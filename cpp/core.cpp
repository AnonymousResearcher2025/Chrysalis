// Chrysalis core: layered proximity graph with DiskANN pruning, immutable epoch
// views with lock-free switching, versioned overlays, interval kernels, and the
// interval edge auditor.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using idx_t = int32_t;
using farr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using u8arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using u16arr = py::array_t<uint16_t, py::array::c_style | py::array::forcecast>;
constexpr float INF = std::numeric_limits<float>::infinity();
constexpr int BLOCK = 1024;

// Fixed summation order: results are bitwise reproducible for a given width,
// which keeps on-the-fly mapping and rotated vectors identical.
static inline float l2sq(const float* a, const float* b, int d) {
  float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int i = 0;
  for (; i + 8 <= d; i += 8)
    for (int j = 0; j < 8; ++j) {
      float t = a[i + j] - b[i + j];
      s[j] += t * t;
    }
  float r = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
  for (; i < d; ++i) {
    float t = a[i] - b[i];
    r += t * t;
  }
  return r;
}

using Cand = std::pair<float, idx_t>;
struct Farther {
  bool operator()(const Cand& a, const Cand& b) const { return a.first > b.first || (a.first == b.first && a.second > b.second); }
};
struct Closer {
  bool operator()(const Cand& a, const Cand& b) const { return a.first < b.first || (a.first == b.first && a.second < b.second); }
};

struct Visited {
  std::vector<uint32_t> tag;
  uint32_t cur = 0;
  void reset(size_t n) {
    if (tag.size() < n) tag.assign(n + n / 4 + 1024, 0), cur = 0;
    if (++cur == 0) std::fill(tag.begin(), tag.end(), 0), cur = 1;
  }
  bool test_set(idx_t i) {
    if (tag[i] == cur) return true;
    tag[i] = cur;
    return false;
  }
};
static thread_local Visited tls_visited;

// ---------------------------------------------------------------- maps
struct Map {
  int kind = 0;  // 0: low-rank affine U V^T x + b, 1: dense x W + b
  int din = 0, dout = 0, rank = 0;
  std::vector<float> V, U, W, b;
  void apply(const float* x, float* y) const {
    if (kind == 1) {
      for (int k = 0; k < dout; ++k) y[k] = b[k];
      for (int i = 0; i < din; ++i) {
        const float xi = x[i];
        const float* row = &W[(size_t)i * dout];
        for (int k = 0; k < dout; ++k) y[k] += xi * row[k];
      }
      return;
    }
    thread_local std::vector<float> z;
    z.assign(rank, 0.f);
    for (int i = 0; i < din; ++i) {
      const float xi = x[i];
      const float* row = &V[(size_t)i * rank];
      for (int j = 0; j < rank; ++j) z[j] += xi * row[j];
    }
    for (int k = 0; k < dout; ++k) {
      const float* row = &U[(size_t)k * rank];
      float s = b[k];
      for (int j = 0; j < rank; ++j) s += row[j] * z[j];
      y[k] = s;
    }
  }
};

static std::shared_ptr<Map> make_lowrank(farr V, farr U, farr b) {
  auto m = std::make_shared<Map>();
  m->kind = 0;
  m->din = (int)V.shape(0);
  m->rank = (int)V.shape(1);
  m->dout = (int)U.shape(0);
  if ((int)U.shape(1) != m->rank || (int)b.size() != m->dout) throw std::invalid_argument("map shapes");
  m->V.assign(V.data(), V.data() + V.size());
  m->U.assign(U.data(), U.data() + U.size());
  m->b.assign(b.data(), b.data() + b.size());
  return m;
}

static std::shared_ptr<Map> make_dense(farr W, farr b) {
  auto m = std::make_shared<Map>();
  m->kind = 1;
  m->din = (int)W.shape(0);
  m->dout = (int)W.shape(1);
  if ((int)b.size() != m->dout) throw std::invalid_argument("map shapes");
  m->W.assign(W.data(), W.data() + W.size());
  m->b.assign(b.data(), b.data() + b.size());
  return m;
}

using MapSet = std::vector<std::shared_ptr<Map>>;

// ---------------------------------------------------------------- graph
// Mutable layered graph used to build indexes and by the baselines.
class Graph {
 public:
  int dim, R, Ru, efc;
  float a2;
  double mL;
  std::vector<float> X;
  std::vector<std::vector<std::vector<idx_t>>> links;
  std::vector<int8_t> lvl;
  idx_t n = 0, entry = -1;
  int maxl = -1;
  std::vector<std::unique_ptr<std::mutex>> mu;
  std::mutex emu;
  std::mt19937_64 rng;

  Graph(int dim, int max_degree, int ef_construction, float alpha, uint64_t seed)
      : dim(dim), R(max_degree), Ru(std::max(2, max_degree / 2)), efc(ef_construction), a2(alpha * alpha),
        mL(1.0 / std::log((double)std::max(2, max_degree / 2))), rng(seed) {}

  const float* vec(idx_t i) const { return &X[(size_t)i * dim]; }
  int degree(int layer) const { return layer == 0 ? R : Ru; }

  void grow(idx_t cap) {
    X.resize((size_t)cap * dim);
    links.resize(cap);
    lvl.resize(cap, 0);
    while ((idx_t)mu.size() < cap) mu.emplace_back(new std::mutex());
  }

  int draw_level() {
    std::uniform_real_distribution<double> u(std::numeric_limits<double>::min(), 1.0);
    return std::min(16, (int)std::floor(-std::log(u(rng)) * mL));
  }

  std::vector<idx_t> nbrs(idx_t i, int l) {
    std::lock_guard<std::mutex> g(*mu[i]);
    if ((int)links[i].size() <= l) return {};
    return links[i][l];
  }

  void greedy(const float* q, idx_t& cur, float& dcur, int l) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (idx_t e : nbrs(cur, l)) {
        float d = l2sq(q, vec(e), dim);
        if (d < dcur || (d == dcur && e < cur)) dcur = d, cur = e, changed = true;
      }
    }
  }

  std::vector<Cand> search_layer(const float* q, idx_t ep, int ef, int l, idx_t skip = -1) {
    Visited& vis = tls_visited;
    vis.reset(n);
    std::priority_queue<Cand, std::vector<Cand>, Farther> cand;
    std::priority_queue<Cand, std::vector<Cand>, Closer> top;
    float d0 = l2sq(q, vec(ep), dim);
    vis.test_set(ep);
    cand.push({d0, ep});
    if (ep != skip) top.push({d0, ep});
    while (!cand.empty()) {
      Cand c = cand.top();
      if ((int)top.size() >= ef && c.first > top.top().first) break;
      cand.pop();
      for (idx_t e : nbrs(c.second, l)) {
        if (vis.test_set(e)) continue;
        float d = l2sq(q, vec(e), dim);
        if ((int)top.size() < ef || d < top.top().first) {
          cand.push({d, e});
          if (e != skip) {
            top.push({d, e});
            if ((int)top.size() > ef) top.pop();
          }
        }
      }
    }
    std::vector<Cand> out;
    out.reserve(top.size());
    while (!top.empty()) out.push_back(top.top()), top.pop();
    std::reverse(out.begin(), out.end());
    return out;
  }

  std::vector<idx_t> prune(idx_t p, const std::vector<Cand>& cands, int M) const {
    std::vector<idx_t> res;
    for (const Cand& c : cands) {
      if (c.second == p) continue;
      if ((int)res.size() >= M) break;
      bool keep = true;
      for (idx_t s : res)
        if (a2 * l2sq(vec(s), vec(c.second), dim) <= c.first) {
          keep = false;
          break;
        }
      if (keep) res.push_back(c.second);
    }
    return res;
  }

  void connect(idx_t i, int l, const std::vector<idx_t>& sel) {
    {
      std::lock_guard<std::mutex> g(*mu[i]);
      links[i][l] = sel;
    }
    int M = degree(l);
    for (idx_t e : sel) {
      std::lock_guard<std::mutex> g(*mu[e]);
      if ((int)links[e].size() <= l) continue;
      auto& L = links[e][l];
      if (std::find(L.begin(), L.end(), i) != L.end()) continue;
      if ((int)L.size() < M) {
        L.push_back(i);
        continue;
      }
      std::vector<Cand> c;
      c.reserve(L.size() + 1);
      for (idx_t x : L) c.push_back({l2sq(vec(e), vec(x), dim), x});
      c.push_back({l2sq(vec(e), vec(i), dim), i});
      std::sort(c.begin(), c.end(), Closer());
      L = prune(e, c, M);
    }
  }

  void link_node(idx_t i, bool relink) {
    const float* q = vec(i);
    int l = lvl[i];
    idx_t ep;
    int L;
    {
      std::lock_guard<std::mutex> g(emu);
      ep = entry;
      L = maxl;
      if (ep < 0) {
        entry = i, maxl = l;
        return;
      }
    }
    if (relink && ep == i) {
      for (int lc = std::min(l, L); lc >= 0; --lc) {
        auto nb = nbrs(i, lc);
        if (!nb.empty()) {
          ep = nb[0], L = lc;
          break;
        }
      }
      if (ep == i) return;
    }
    idx_t cur = ep;
    float dcur = l2sq(q, vec(cur), dim);
    for (int lc = L; lc > l; --lc) greedy(q, cur, dcur, lc);
    for (int lc = std::min(l, L); lc >= 0; --lc) {
      auto W = search_layer(q, cur, efc, lc, i);
      if (W.empty()) continue;
      connect(i, lc, prune(i, W, degree(lc)));
      cur = W[0].second;
    }
    if (!relink && l > L) {
      std::lock_guard<std::mutex> g(emu);
      if (l > maxl) maxl = l, entry = i;
    }
  }

  void build(farr data, int threads) {
    if ((int)data.shape(1) != dim) throw std::invalid_argument("dimension");
    idx_t m = (idx_t)data.shape(0);
    idx_t base = n;
    grow(base + m);
    std::memcpy(&X[(size_t)base * dim], data.data(), sizeof(float) * (size_t)m * dim);
    for (idx_t i = base; i < base + m; ++i) {
      lvl[i] = (int8_t)draw_level();
      links[i].assign(lvl[i] + 1, {});
    }
    n = base + m;
    py::gil_scoped_release nogil;
    idx_t start = base;
    if (entry < 0 && m > 0) link_node(start++, false);
    std::atomic<idx_t> next(start);
    auto work = [&]() {
      for (idx_t i = next++; i < base + m; i = next++) link_node(i, false);
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < std::max(1, threads); ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
  }

  idx_t insert(farr v) {
    if ((int)v.size() != dim) throw std::invalid_argument("dimension");
    idx_t i = n;
    grow(n + 1);
    std::memcpy(&X[(size_t)i * dim], v.data(), sizeof(float) * dim);
    lvl[i] = (int8_t)draw_level();
    links[i].assign(lvl[i] + 1, {});
    n = i + 1;
    link_node(i, false);
    return i;
  }

  void reinsert(idx_t i, farr v) {
    if ((int)v.size() != dim || i < 0 || i >= n) throw std::invalid_argument("reinsert");
    {
      std::lock_guard<std::mutex> g(*mu[i]);
      std::memcpy(&X[(size_t)i * dim], v.data(), sizeof(float) * dim);
    }
    link_node(i, true);
  }

  std::pair<std::vector<idx_t>, std::vector<float>> search_one(const float* q, int ef) {
    if (entry < 0) return {};
    idx_t cur = entry;
    float dcur = l2sq(q, vec(cur), dim);
    for (int l = maxl; l > 0; --l) greedy(q, cur, dcur, l);
    auto W = search_layer(q, cur, ef, 0);
    std::vector<idx_t> ids;
    std::vector<float> ds;
    for (auto& c : W) ids.push_back(c.second), ds.push_back(std::sqrt(c.first));
    return {ids, ds};
  }

  py::tuple search(farr q, int ef) {
    std::pair<std::vector<idx_t>, std::vector<float>> r;
    {
      py::gil_scoped_release nogil;
      r = search_one(q.data(), ef);
    }
    return py::make_tuple(py::array(py::cast(r.first)), py::array(py::cast(r.second)));
  }

  std::shared_ptr<Graph> with_vectors(farr data) const {
    if ((idx_t)data.shape(0) != n) throw std::invalid_argument("row count");
    auto g = std::make_shared<Graph>((int)data.shape(1), R, efc, std::sqrt(a2), 0);
    g->grow(n);
    g->n = n;
    std::memcpy(g->X.data(), data.data(), sizeof(float) * data.size());
    for (idx_t i = 0; i < n; ++i) g->links[i] = links[i], g->lvl[i] = lvl[i];
    g->entry = entry;
    g->maxl = maxl;
    g->rng = rng;
    return g;
  }

  farr vectors() const {
    farr out({(py::ssize_t)n, (py::ssize_t)dim});
    std::memcpy(out.mutable_data(), X.data(), sizeof(float) * (size_t)n * dim);
    return out;
  }

  std::vector<idx_t> neighbors(idx_t i, int l) { return nbrs(i, l); }

  // Layer-0 adjacency as fixed rows: [count, neighbors..., -1 padding].
  iarr layer0() const {
    iarr out({(py::ssize_t)n, (py::ssize_t)(R + 1)});
    int32_t* o = out.mutable_data();
    std::fill(o, o + (size_t)n * (R + 1), -1);
    for (idx_t i = 0; i < n; ++i) {
      auto& L = links[i][0];
      o[(size_t)i * (R + 1)] = (int32_t)L.size();
      std::copy(L.begin(), L.end(), o + (size_t)i * (R + 1) + 1);
    }
    return out;
  }

  py::tuple upper() const {
    py::list lists;
    for (idx_t i = 0; i < n; ++i)
      for (int l = 1; l < (int)links[i].size(); ++l) lists.append(py::make_tuple(l, i, links[i][l]));
    return py::make_tuple(py::array(py::cast(lvl)), lists, entry, maxl);
  }

  static std::shared_ptr<Graph> from_adjacency(int dim, int R, int efc, float alpha, iarr rows, py::array_t<int8_t> levels,
                                               std::vector<std::tuple<int, idx_t, std::vector<idx_t>>> up, idx_t entry,
                                               int maxl) {
    if ((int)rows.shape(1) != R + 1 || levels.size() != rows.shape(0)) throw std::invalid_argument("adjacency shape");
    auto g = std::make_shared<Graph>(dim, R, efc, alpha, 0);
    idx_t n = (idx_t)rows.shape(0);
    g->grow(n);
    g->n = n, g->entry = entry, g->maxl = maxl;
    for (idx_t i = 0; i < n; ++i) {
      g->lvl[i] = levels.data()[i];
      g->links[i].assign(g->lvl[i] + 1, {});
      const int32_t* r = rows.data() + (size_t)i * (R + 1);
      g->links[i][0].assign(r + 1, r + 1 + r[0]);
    }
    for (auto& t : up) g->links[std::get<1>(t)][std::get<0>(t)] = std::get<2>(t);
    return g;
  }

  // Without vectors the file holds only the adjacency (vectors then live in index segments).
  void save(const std::string& path, bool with_vectors) const {
    std::ofstream f(path, std::ios::binary);
    int32_t hdr[7] = {dim, R, efc, n, entry, maxl, with_vectors ? 1 : 0};
    f.write((const char*)hdr, sizeof(hdr));
    float a = std::sqrt(a2);
    f.write((const char*)&a, sizeof(a));
    if (with_vectors) f.write((const char*)X.data(), sizeof(float) * (size_t)n * dim);
    f.write((const char*)lvl.data(), n);
    for (idx_t i = 0; i < n; ++i)
      for (auto& L : links[i]) {
        int32_t c = (int32_t)L.size();
        f.write((const char*)&c, 4);
        f.write((const char*)L.data(), 4 * c);
      }
    f.flush();
    if (!f) throw std::runtime_error("graph write failed");
  }

  static std::shared_ptr<Graph> load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int32_t hdr[7];
    float a;
    f.read((char*)hdr, sizeof(hdr));
    f.read((char*)&a, sizeof(a));
    auto g = std::make_shared<Graph>(hdr[0], hdr[1], hdr[2], a, 0);
    g->grow(hdr[3]);
    g->n = hdr[3];
    g->entry = hdr[4];
    g->maxl = hdr[5];
    if (hdr[6]) f.read((char*)g->X.data(), sizeof(float) * (size_t)g->n * g->dim);
    f.read((char*)g->lvl.data(), g->n);
    for (idx_t i = 0; i < g->n; ++i) {
      g->links[i].resize(g->lvl[i] + 1);
      for (auto& L : g->links[i]) {
        int32_t c;
        f.read((char*)&c, 4);
        L.resize(c);
        f.read((char*)L.data(), 4 * c);
      }
    }
    if (!f) throw std::runtime_error("graph read failed");
    return g;
  }
};

// ---------------------------------------------------------------- storage objects
struct Segment {
  int width;
  bool legacy;
  idx_t rows;
  std::vector<float> data;
  Segment(int width, bool legacy, farr a) : width(width), legacy(legacy), rows((idx_t)a.shape(0)) {
    if ((int)a.shape(1) != width) throw std::invalid_argument("segment width");
    data.assign(a.data(), a.data() + a.size());
  }
  // Callers guarantee that no live view reads these rows (see MigratingIndex).
  void write_rows(iarr local, farr v) {
    if ((int)v.shape(1) != width || v.shape(0) != local.size()) throw std::invalid_argument("rows");
    for (py::ssize_t r = 0; r < local.size(); ++r) {
      idx_t j = local.data()[r];
      if (j < 0 || j >= rows) throw std::out_of_range("row");
      std::memcpy(&data[(size_t)j * width], v.data() + (size_t)r * width, sizeof(float) * width);
    }
  }
  farr array() const {
    farr out({(py::ssize_t)rows, (py::ssize_t)width});
    std::memcpy(out.mutable_data(), data.data(), sizeof(float) * data.size());
    return out;
  }
};

// Exact vectors of one model version published during an epoch. Readers never lock.
struct Overlay {
  int dim;
  idx_t cap;
  uint64_t version;
  std::unique_ptr<std::atomic<const float*>[]> slot;
  std::unique_ptr<std::atomic<uint8_t>[]> dead;
  std::mutex mu;
  std::vector<std::unique_ptr<float[]>> owned;
  std::atomic<int64_t> count{0};
  Overlay(idx_t cap, int dim, uint64_t version)
      : dim(dim), cap(cap), version(version), slot(new std::atomic<const float*>[cap]), dead(new std::atomic<uint8_t>[cap]) {
    for (idx_t i = 0; i < cap; ++i) slot[i].store(nullptr), dead[i].store(0);
  }
  const float* get(idx_t i) const { return (i >= 0 && i < cap) ? slot[i].load(std::memory_order_acquire) : nullptr; }
  bool put(idx_t i, farr v) {
    if ((int)v.size() != dim) throw std::invalid_argument("overlay width");
    if (i < 0 || i >= cap) throw std::out_of_range("overlay capacity");
    std::lock_guard<std::mutex> g(mu);
    if (slot[i].load()) return false;
    std::unique_ptr<float[]> p(new float[dim]);
    std::memcpy(p.get(), v.data(), sizeof(float) * dim);
    slot[i].store(p.get(), std::memory_order_release);
    owned.push_back(std::move(p));
    ++count;
    return true;
  }
  py::object get_py(idx_t i) const {
    const float* p = get(i);
    if (!p) return py::none();
    farr out(dim);
    std::memcpy(out.mutable_data(), p, sizeof(float) * dim);
    return out;
  }
  void kill(idx_t i) { dead[i].store(1, std::memory_order_release); }
  bool is_dead(idx_t i) const { return i >= 0 && i < cap && dead[i].load(std::memory_order_acquire); }
  py::tuple items() {
    std::vector<idx_t> ids;
    for (idx_t i = 0; i < cap; ++i)
      if (slot[i].load(std::memory_order_acquire)) ids.push_back(i);
    farr v({(py::ssize_t)ids.size(), (py::ssize_t)dim});
    for (size_t r = 0; r < ids.size(); ++r) std::memcpy(v.mutable_data() + r * dim, get(ids[r]), sizeof(float) * dim);
    return py::make_tuple(py::array(py::cast(ids)), v);
  }
};

// ---------------------------------------------------------------- views
enum : uint8_t { F_NATIVE = 1, F_REDIRECT = 2, F_DELETED = 4 };

struct AdjBlock {
  int R;
  std::vector<idx_t> nbr;
  std::vector<uint16_t> cnt;
  explicit AdjBlock(int R) : R(R), nbr((size_t)BLOCK * R, -1), cnt(BLOCK, 0) {}
};

struct Upper {
  idx_t entry = -1;
  int maxl = -1;
  std::vector<std::unordered_map<idx_t, std::vector<idx_t>>> layers;  // layers[l - 1]
};

struct View {
  uint64_t epoch = 0, version = 0;
  int dim = 0, R = 64, seg_size = 4096;
  idx_t n = 0;
  std::vector<std::shared_ptr<const AdjBlock>> l0;
  std::shared_ptr<const Upper> upper;
  std::vector<std::shared_ptr<Segment>> segs;
  std::shared_ptr<const std::vector<uint16_t>> region;
  std::shared_ptr<const std::vector<uint8_t>> flags;
  std::shared_ptr<Overlay> frozen, live;
  std::shared_ptr<const MapSet> maps;
  mutable std::once_flag rev_once;
  mutable std::vector<idx_t> rev_off, rev;

  std::shared_ptr<View> copy() const {
    auto v = std::make_shared<View>();
    v->epoch = epoch, v->version = version, v->dim = dim, v->R = R, v->seg_size = seg_size, v->n = n;
    v->l0 = l0, v->upper = upper, v->segs = segs, v->region = region, v->flags = flags;
    v->frozen = frozen, v->live = live, v->maps = maps;
    return v;
  }

  const float* routing(idx_t i, float* buf) const {
    uint8_t f = (*flags)[i];
    if (f & F_REDIRECT) return frozen->get(i);
    const Segment& s = *segs[i / seg_size];
    const float* row = s.data.data() + (size_t)(i % seg_size) * s.width;
    if (!s.legacy) return row;
    (*maps)[(*region)[i]]->apply(row, buf);
    return buf;
  }

  float dist2(const float* q, idx_t i) const {
    thread_local std::vector<float> buf;
    if ((int)buf.size() < dim) buf.resize(dim);
    return l2sq(q, routing(i, buf.data()), dim);
  }

  std::pair<const idx_t*, int> adj(idx_t i, int l) const {
    if (l == 0) {
      const AdjBlock& b = *l0[i / BLOCK];
      int li = i % BLOCK;
      return {&b.nbr[(size_t)li * b.R], b.cnt[li]};
    }
    if (l > upper->maxl) return {nullptr, 0};
    auto& m = upper->layers[l - 1];
    auto it = m.find(i);
    if (it == m.end()) return {nullptr, 0};
    return {it->second.data(), (int)it->second.size()};
  }

  bool excluded(idx_t i) const { return ((*flags)[i] & F_DELETED) || (live && live->is_dead(i)); }

  std::vector<Cand> search_one(const float* q, int ef) const {
    std::vector<Cand> out;
    idx_t cur = upper->entry;
    if (cur < 0 || n == 0) return out;
    float dcur = dist2(q, cur);
    for (int l = upper->maxl; l > 0; --l) {
      bool changed = true;
      while (changed) {
        changed = false;
        auto a = adj(cur, l);
        for (int j = 0; j < a.second; ++j) {
          idx_t e = a.first[j];
          float d = dist2(q, e);
          if (d < dcur || (d == dcur && e < cur)) dcur = d, cur = e, changed = true;
        }
      }
    }
    Visited& vis = tls_visited;
    vis.reset(n);
    std::priority_queue<Cand, std::vector<Cand>, Farther> cand;
    std::priority_queue<Cand, std::vector<Cand>, Closer> top;
    vis.test_set(cur);
    cand.push({dcur, cur});
    if (!excluded(cur)) top.push({dcur, cur});
    while (!cand.empty()) {
      Cand c = cand.top();
      if ((int)top.size() >= ef && c.first > top.top().first) break;
      cand.pop();
      auto a = adj(c.second, 0);
      for (int j = 0; j < a.second; ++j) {
        idx_t e = a.first[j];
        if (e >= n || vis.test_set(e)) continue;
        float d = dist2(q, e);
        if ((int)top.size() < ef || d < top.top().first) {
          cand.push({d, e});
          if (!excluded(e)) {
            top.push({d, e});
            if ((int)top.size() > ef) top.pop();
          }
        }
      }
    }
    while (!top.empty()) out.push_back(top.top()), top.pop();
    std::reverse(out.begin(), out.end());
    for (auto& c : out) c.first = std::sqrt(c.first);
    return out;
  }

  py::tuple search(farr q, int ef) const {
    if ((int)q.size() != dim) throw std::invalid_argument("query width");
    std::vector<Cand> r;
    {
      py::gil_scoped_release nogil;
      r = search_one(q.data(), ef);
    }
    iarr ids((py::ssize_t)r.size());
    farr ds((py::ssize_t)r.size());
    for (size_t j = 0; j < r.size(); ++j) ids.mutable_data()[j] = r[j].second, ds.mutable_data()[j] = r[j].first;
    return py::make_tuple(ids, ds);
  }

  // Exact flags and distances: in-view natives keep their routing distance;
  // vectors published to the live overlay give exact distances for ranking.
  py::tuple annotate(farr q, iarr ids, farr d) const {
    py::ssize_t m = ids.size();
    u8arr exact(m);
    farr out(m);
    for (py::ssize_t j = 0; j < m; ++j) {
      idx_t i = ids.data()[j];
      float dj = d.data()[j];
      uint8_t e = 0;
      if ((*flags)[i] & F_NATIVE) {
        e = 1;
      } else if (live && live->version == version) {
        const float* v = live->get(i);
        if (v) e = 1, dj = std::sqrt(l2sq(q.data(), v, dim));
      }
      exact.mutable_data()[j] = e;
      out.mutable_data()[j] = dj;
    }
    return py::make_tuple(exact, out);
  }

  farr rotate_segment(int s) const {
    const Segment& seg = *segs[s];
    farr out({(py::ssize_t)seg_size, (py::ssize_t)dim});
    float* o = out.mutable_data();
    idx_t base = (idx_t)s * seg_size;
    for (idx_t j = 0; j < seg_size; ++j) {
      idx_t i = base + j;
      float* y = o + (size_t)j * dim;
      if (i >= n || j >= seg.rows) {
        std::fill(y, y + dim, 0.f);
      } else if ((*flags)[i] & F_REDIRECT) {
        std::memcpy(y, frozen->get(i), sizeof(float) * dim);
      } else if (seg.legacy) {
        (*maps)[(*region)[i]]->apply(&seg.data[(size_t)j * seg.width], y);
      } else {
        std::memcpy(y, &seg.data[(size_t)j * seg.width], sizeof(float) * dim);
      }
    }
    return out;
  }

  farr routing_vectors(iarr ids) const {
    farr out({(py::ssize_t)ids.size(), (py::ssize_t)dim});
    std::vector<float> buf(dim);
    for (py::ssize_t j = 0; j < ids.size(); ++j)
      std::memcpy(out.mutable_data() + j * dim, routing(ids.data()[j], buf.data()), sizeof(float) * dim);
    return out;
  }

  // Same view with segment s replaced and its redirects cleared; routing values are unchanged.
  std::shared_ptr<View> with_segment(int s, std::shared_ptr<Segment> seg) const {
    if (seg->legacy || seg->width != dim) throw std::invalid_argument("rotated segment");
    auto v = copy();
    v->segs[s] = seg;
    auto f = std::make_shared<std::vector<uint8_t>>(*flags);
    idx_t base = (idx_t)s * seg_size;
    for (idx_t j = 0; j < seg->rows && base + j < n; ++j) (*f)[base + j] &= (uint8_t)~F_REDIRECT;
    v->flags = f;
    return v;
  }

  std::shared_ptr<View> with_flags(u8arr f, std::shared_ptr<Overlay> fr) const {
    if ((idx_t)f.size() != n) throw std::invalid_argument("flags");
    auto v = copy();
    v->flags = std::make_shared<std::vector<uint8_t>>(f.data(), f.data() + f.size());
    v->frozen = fr;
    return v;
  }

  std::shared_ptr<View> next(uint64_t ep, const std::unordered_map<idx_t, std::vector<idx_t>>& c0,
                             const std::vector<std::tuple<int, idx_t, std::vector<idx_t>>>& cu, u8arr f, u16arr reg,
                             std::vector<std::shared_ptr<Segment>> extra, std::shared_ptr<Overlay> fr,
                             std::shared_ptr<Overlay> lv) const {
    auto v = copy();
    v->epoch = ep;
    idx_t nn = (idx_t)f.size();
    if (nn < n || (idx_t)reg.size() != nn) throw std::invalid_argument("node arrays");
    v->n = nn;
    for (auto& s : extra) v->segs.push_back(s);
    if ((idx_t)v->segs.size() * seg_size < nn) throw std::invalid_argument("segments");
    while ((idx_t)v->l0.size() * BLOCK < nn) v->l0.push_back(std::make_shared<AdjBlock>(R));
    std::unordered_map<size_t, std::shared_ptr<AdjBlock>> fresh;
    for (auto& kv : c0) {
      idx_t i = kv.first;
      if (i < 0 || i >= nn || (int)kv.second.size() > R) throw std::invalid_argument("staged edges");
      size_t b = i / BLOCK;
      auto it = fresh.find(b);
      if (it == fresh.end()) it = fresh.emplace(b, std::make_shared<AdjBlock>(*v->l0[b])).first;
      AdjBlock& blk = *it->second;
      int li = i % BLOCK;
      std::copy(kv.second.begin(), kv.second.end(), blk.nbr.begin() + (size_t)li * R);
      blk.cnt[li] = (uint16_t)kv.second.size();
    }
    for (auto& kv : fresh) v->l0[kv.first] = kv.second;
    if (!cu.empty()) {
      auto u = std::make_shared<Upper>(*upper);
      for (auto& t : cu) {
        int l = std::get<0>(t);
        if (l < 1 || l > u->maxl) throw std::invalid_argument("upper layer");
        u->layers[l - 1][std::get<1>(t)] = std::get<2>(t);
      }
      v->upper = u;
    }
    v->flags = std::make_shared<std::vector<uint8_t>>(f.data(), f.data() + nn);
    v->region = std::make_shared<std::vector<uint16_t>>(reg.data(), reg.data() + nn);
    v->frozen = fr;
    v->live = lv;
    if (lv && lv->cap < nn) throw std::invalid_argument("overlay capacity");
    if ((lv && lv->version != version) || (fr && fr->version != version)) throw std::invalid_argument("overlay version");
    return v;
  }

  void build_reverse() const {
    std::call_once(rev_once, [this]() {
      rev_off.assign(n + 1, 0);
      for (idx_t i = 0; i < n; ++i) {
        auto a = adj(i, 0);
        for (int j = 0; j < a.second; ++j) rev_off[a.first[j] + 1]++;
      }
      for (idx_t i = 0; i < n; ++i) rev_off[i + 1] += rev_off[i];
      rev.assign(rev_off[n], 0);
      std::vector<idx_t> pos(rev_off.begin(), rev_off.end() - 1);
      for (idx_t i = 0; i < n; ++i) {
        auto a = adj(i, 0);
        for (int j = 0; j < a.second; ++j) rev[pos[a.first[j]]++] = i;
      }
    });
  }

  std::vector<idx_t> in_neighbors(idx_t i) const {
    build_reverse();
    return std::vector<idx_t>(rev.begin() + rev_off[i], rev.begin() + rev_off[i + 1]);
  }

  iarr indegrees() const {
    iarr out(n);
    std::fill(out.mutable_data(), out.mutable_data() + n, 0);
    for (idx_t i = 0; i < n; ++i) {
      auto a = adj(i, 0);
      for (int j = 0; j < a.second; ++j) out.mutable_data()[a.first[j]]++;
    }
    return out;
  }

  std::vector<idx_t> neighbors(idx_t i, int l) const {
    auto a = adj(i, l);
    return std::vector<idx_t>(a.first, a.first + a.second);
  }

  py::dict upper_layers() const {
    py::dict d;
    d["entry"] = upper->entry;
    d["maxl"] = upper->maxl;
    py::list layers;
    for (auto& m : upper->layers) layers.append(py::cast(m));
    d["layers"] = layers;
    return d;
  }

  static std::shared_ptr<View> initial(const Graph& g, int seg_size, std::vector<std::shared_ptr<Segment>> segs, u16arr reg,
                                       MapSet maps, u8arr f, std::shared_ptr<Overlay> fr, std::shared_ptr<Overlay> lv,
                                       int dim, uint64_t version) {
    if ((lv && lv->version != version) || (fr && fr->version != version)) throw std::invalid_argument("overlay version");
    auto v = std::make_shared<View>();
    v->version = version, v->dim = dim, v->R = g.R, v->seg_size = seg_size, v->n = g.n;
    if ((idx_t)reg.size() != g.n || (idx_t)f.size() != g.n) throw std::invalid_argument("node arrays");
    if ((idx_t)segs.size() * seg_size < g.n) throw std::invalid_argument("segments");
    for (idx_t b = 0; b * BLOCK < g.n; ++b) {
      auto blk = std::make_shared<AdjBlock>(g.R);
      for (idx_t i = b * BLOCK; i < std::min(g.n, (b + 1) * BLOCK); ++i) {
        auto& L = g.links[i][0];
        int li = i % BLOCK;
        std::copy(L.begin(), L.end(), blk->nbr.begin() + (size_t)li * g.R);
        blk->cnt[li] = (uint16_t)L.size();
      }
      v->l0.push_back(blk);
    }
    auto u = std::make_shared<Upper>();
    u->entry = g.entry, u->maxl = g.maxl;
    u->layers.resize(std::max(0, g.maxl));
    for (idx_t i = 0; i < g.n; ++i)
      for (int l = 1; l <= g.lvl[i]; ++l) u->layers[l - 1][i] = g.links[i][l];
    v->upper = u;
    v->segs = std::move(segs);
    v->region = std::make_shared<std::vector<uint16_t>>(reg.data(), reg.data() + reg.size());
    v->flags = std::make_shared<std::vector<uint8_t>>(f.data(), f.data() + f.size());
    v->frozen = fr, v->live = lv;
    v->maps = std::make_shared<MapSet>(std::move(maps));
    return v;
  }
};

// Two-slot holder. Readers pin the current view without locks; publish installs
// a new view and returns once no reader still holds the previous one.
struct Holder {
  std::atomic<uint64_t> ver{0};
  std::shared_ptr<const View> slot[2];
  std::atomic<int64_t> readers[2];
  std::mutex writer;
  Holder() { readers[0] = 0, readers[1] = 0; }

  std::pair<int, const View*> acquire() {
    for (;;) {
      uint64_t v = ver.load(std::memory_order_seq_cst);
      int s = (int)(v & 1);
      readers[s].fetch_add(1, std::memory_order_seq_cst);
      if (ver.load(std::memory_order_seq_cst) == v && slot[s]) return {s, slot[s].get()};
      readers[s].fetch_sub(1, std::memory_order_seq_cst);
      if (!slot[s] && ver.load() == v) throw std::runtime_error("no view published");
    }
  }
  void release(int s) { readers[s].fetch_sub(1, std::memory_order_seq_cst); }

  void publish(std::shared_ptr<const View> v) {
    py::gil_scoped_release nogil;
    std::lock_guard<std::mutex> g(writer);
    uint64_t cur = ver.load();
    int nx = (int)((cur + 1) & 1), old = (int)(cur & 1);
    while (readers[nx].load(std::memory_order_seq_cst) != 0) std::this_thread::yield();
    slot[nx] = std::move(v);
    ver.store(cur + 1, std::memory_order_seq_cst);
    while (readers[old].load(std::memory_order_seq_cst) != 0) std::this_thread::yield();
    slot[old].reset();
  }

  std::shared_ptr<const View> current() {
    auto p = acquire();
    std::shared_ptr<const View> out = slot[p.first];
    release(p.first);
    return out;
  }
};

struct Pin {
  Holder* h;
  int s;
  const View* v;
  bool held;
  explicit Pin(Holder& hh) : h(&hh) {
    auto p = hh.acquire();
    s = p.first, v = p.second, held = true;
  }
  ~Pin() { done(); }
  void done() {
    if (held) h->release(s), held = false;
  }
  const View& view() const {
    if (!held) throw std::runtime_error("released pin");
    return *v;
  }
};

// ---------------------------------------------------------------- interval kernels
static void top_k(const float* d, const int32_t* ids, int m, int k, std::vector<int>& order) {
  order.resize(m);
  for (int j = 0; j < m; ++j) order[j] = j;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return d[a] < d[b] || (d[a] == d[b] && ids[a] < ids[b]); });
  order.resize(std::min(m, k));
}

static inline void bounds(float d, bool exact, float w, float& lb, float& ub) {
  if (exact) {
    lb = ub = d;
  } else if (std::isinf(w)) {
    lb = 0.f, ub = INF;
  } else {
    lb = std::max(0.f, d - w), ub = d + w;
  }
}

// Returns (indices of S in C, m, z, bound) for R = max(m, k - z) / k.
static py::tuple certify(farr d, u8arr exact, farr w, iarr ids, int k) {
  int m_c = (int)d.size();
  std::vector<int> S;
  top_k(d.data(), ids.data(), m_c, k, S);
  if (m_c <= k) return py::make_tuple(py::cast(S), (int)S.size(), 0, 1.0);
  std::vector<float> lb(m_c), ub(m_c);
  for (int j = 0; j < m_c; ++j) bounds(d.data()[j], exact.data()[j], w.data()[j], lb[j], ub[j]);
  std::vector<char> in_s(m_c, 0);
  for (int j : S) in_s[j] = 1;
  float min_lb_out = INF;
  for (int j = 0; j < m_c; ++j)
    if (!in_s[j]) min_lb_out = std::min(min_lb_out, lb[j]);
  int m = 0;
  for (int j : S)
    if (ub[j] < min_lb_out) ++m;
  std::vector<float> ubs(ub);
  std::sort(ubs.begin(), ubs.end());
  int z = 0;
  for (int j = 0; j < m_c; ++j) {
    if (in_s[j]) continue;
    int sure = (int)(std::lower_bound(ubs.begin(), ubs.end(), lb[j]) - ubs.begin());
    if (sure < k) ++z;
  }
  double bound = std::max(0, std::max(m, k - z)) / (double)k;
  return py::make_tuple(py::cast(S), m, z, bound);
}

// Bridged candidates whose interval overlaps the k-th result's, largest overlap first.
static std::vector<int> ambiguous(farr d, u8arr exact, farr w, iarr ids, int k) {
  int m_c = (int)d.size();
  std::vector<int> S;
  top_k(d.data(), ids.data(), m_c, k, S);
  std::vector<int> out;
  if ((int)S.size() < k) {
    for (int j = 0; j < m_c; ++j)
      if (!exact.data()[j]) out.push_back(j);
    return out;
  }
  int kk = S.back();
  float lk, uk;
  bounds(d.data()[kk], exact.data()[kk], w.data()[kk], lk, uk);
  std::vector<std::tuple<float, float, idx_t, int>> amb;
  for (int j = 0; j < m_c; ++j) {
    if (exact.data()[j]) continue;
    float lb, ub;
    bounds(d.data()[j], false, w.data()[j], lb, ub);
    float ov = std::min(ub, uk) - std::max(lb, lk);
    if (ov >= 0) amb.emplace_back(-ov, std::fabs(d.data()[j] - d.data()[kk]), ids.data()[j], j);
  }
  std::sort(amb.begin(), amb.end());
  for (auto& t : amb) out.push_back(std::get<3>(t));
  return out;
}

// ---------------------------------------------------------------- edge auditor
// Interval form of DiskANN's rule over one view. Distances between bridged
// endpoints carry the vector radii gamma; native pairs are exact.
class Auditor {
 public:
  std::shared_ptr<const View> v;
  std::vector<float> gamma;
  float alpha;
  std::unordered_map<idx_t, std::vector<idx_t>> staged0;
  std::map<std::pair<int, idx_t>, std::vector<idx_t>> stagedu;
  std::set<std::pair<int, idx_t>> dirty;
  int64_t changed = 0;

  Auditor(std::shared_ptr<const View> view, farr g, float alpha, int pool_limit)
      : v(std::move(view)), alpha(alpha), pool_limit(pool_limit) {
    gamma.assign(g.data(), g.data() + g.size());
  }

  int pool_limit;
  bool native(idx_t i) const { return (i < v->n && ((*v->flags)[i] & F_NATIVE)) || (v->live && v->live->get(i)); }
  const float* best(idx_t i, float* buf) const {
    if (v->live) {
      const float* p = v->live->get(i);
      if (p) return p;
    }
    return v->routing(i, buf);
  }
  float rad(idx_t i) const { return native(i) ? 0.f : gamma[(*v->region)[i]]; }
  bool gone(idx_t i) const {
    if (i < v->n) return v->excluded(i);
    return !v->live || !v->live->get(i) || v->live->is_dead(i);
  }

  std::vector<idx_t> list(idx_t i, int l) const {
    if (l == 0) {
      auto it = staged0.find(i);
      if (it != staged0.end()) return it->second;
    } else {
      auto it = stagedu.find({l, i});
      if (it != stagedu.end()) return it->second;
    }
    if (i >= v->n) return {};
    return v->neighbors(i, l);
  }

  void set(idx_t i, int l, std::vector<idx_t> L) {
    std::vector<idx_t> base = i < v->n ? v->neighbors(i, l) : std::vector<idx_t>();
    auto prev = list(i, l);
    auto diff = [&](const std::vector<idx_t>& a) {
      std::unordered_set<idx_t> A(a.begin(), a.end()), B(base.begin(), base.end());
      int64_t c = 0;
      for (idx_t x : A) c += !B.count(x);
      for (idx_t x : B) c += !A.count(x);
      return c;
    };
    changed += diff(L) - diff(prev);
    dirty.insert({l, i});
    if (l == 0) staged0[i] = std::move(L);
    else stagedu[{l, i}] = std::move(L);
  }

  struct Vec {
    const float* p;
    float r;
  };

  void interval(const Vec& a, const Vec& b, float& lb, float& ub) const {
    float d = std::sqrt(l2sq(a.p, b.p, v->dim));
    float h = a.r + b.r;
    lb = std::max(0.f, d - h), ub = d + h;
  }

  // Returns the pruned list; status-quo for undecided candidates, recorded in `pending`.
  std::vector<idx_t> select(idx_t p, const Vec& pv, const std::vector<idx_t>& pool_in, const std::vector<idx_t>& current, int M,
                            std::vector<std::pair<idx_t, idx_t>>& pending) const {
    std::unordered_set<idx_t> cur(current.begin(), current.end());
    std::vector<float> tmp(v->dim);
    std::vector<std::pair<float, idx_t>> order;
    order.reserve(pool_in.size());
    for (idx_t x : pool_in) order.emplace_back(l2sq(pv.p, best(x, tmp.data()), v->dim), x);
    std::sort(order.begin(), order.end());
    if ((int)order.size() > pool_limit) order.resize(pool_limit);
    size_t P = order.size();
    std::vector<float> store(P * v->dim);
    std::vector<Vec> pvv(P);
    for (size_t j = 0; j < P; ++j) {
      const float* b = best(order[j].second, &store[j * v->dim]);
      pvv[j] = {b, rad(order[j].second)};
    }
    std::vector<size_t> kept;
    for (size_t j = 0; j < P; ++j) {
      if ((int)kept.size() >= M) break;
      float lpc, upc;
      interval(pv, pvv[j], lpc, upc);
      int decision = 1;  // 1 keep, 0 prune, 2 undecided
      for (size_t s : kept) {
        float lsc, usc;
        interval(pvv[s], pvv[j], lsc, usc);
        if (alpha * usc <= lpc) {
          decision = 0;
          break;
        }
        if (!(alpha * lsc > upc)) decision = 2;
      }
      idx_t c = order[j].second;
      if (decision == 1) {
        kept.push_back(j);
      } else if (decision == 2) {
        pending.emplace_back(p, c);
        if (cur.count(c)) kept.push_back(j);
      }
    }
    std::vector<idx_t> out;
    for (size_t j : kept) out.push_back(order[j].second);
    return out;
  }

  void add_reverse(idx_t p, const Vec& pv, int l, const std::vector<idx_t>& sel, std::vector<std::pair<idx_t, idx_t>>& pending) {
    int M = l == 0 ? v->R : std::max(2, v->R / 2);
    for (idx_t q : sel) {
      auto L = list(q, l);
      if (std::find(L.begin(), L.end(), p) != L.end()) continue;
      std::vector<idx_t> pool;
      for (idx_t x : L)
        if (!gone(x)) pool.push_back(x);
      if ((int)pool.size() < M) {
        pool.push_back(p);
        set(q, l, pool);
        continue;
      }
      std::vector<float> qb(v->dim);
      Vec qv{best(q, qb.data()), rad(q)};
      pool.push_back(p);
      std::vector<idx_t> current(L);
      auto next = select(q, qv, pool, current, M, pending);
      if (next != L) set(q, l, next);
    }
  }

  // Reruns neighbor selection for p over its neighbors, reverse edges and two-hop neighbors.
  py::tuple audit(idx_t p) {
    std::vector<std::pair<idx_t, idx_t>> pending;
    {
      py::gil_scoped_release nogil;
      if (p >= 0 && p < v->n && !gone(p)) audit_one(p, pending);
    }
    return py::make_tuple(pending, changed);
  }

  void audit_one(idx_t p, std::vector<std::pair<idx_t, idx_t>>& pending) {
    v->build_reverse();
    std::vector<float> pb(v->dim);
    Vec pv{best(p, pb.data()), rad(p)};
    int top = 0;
    while (top < v->upper->maxl && !v->neighbors(p, top + 1).empty()) ++top;
    for (int l = 0; l <= top; ++l) {
      int M = l == 0 ? v->R : std::max(2, v->R / 2);
      auto current = list(p, l);
      std::unordered_set<idx_t> pool_set;
      for (idx_t a : current) {
        pool_set.insert(a);
        for (idx_t b : list(a, l)) pool_set.insert(b);
      }
      if (l == 0)
        for (idx_t j = v->rev_off[p]; j < v->rev_off[p + 1]; ++j) pool_set.insert(v->rev[j]);
      std::vector<idx_t> pool;
      for (idx_t x : pool_set)
        if (x != p && !gone(x)) pool.push_back(x);
      std::sort(pool.begin(), pool.end());
      auto sel = select(p, pv, pool, current, M, pending);
      if (sel != current) set(p, l, sel);
      add_reverse(p, pv, l, sel, pending);
    }
  }

  // Neighbor selection for a new native node `id` with vector `x` (inserts).
  py::tuple link(idx_t id, farr x, int ef) {
    std::vector<std::pair<idx_t, idx_t>> pending;
    idx_t nearest = -1;
    {
      py::gil_scoped_release nogil;
      auto C = v->search_one(x.data(), ef);
      std::vector<idx_t> pool;
      for (auto& c : C) pool.push_back(c.second);
      Vec pv{x.data(), 0.f};
      auto sel = select(id, pv, pool, {}, v->R, pending);
      set_new(id, sel);
      add_reverse(id, pv, 0, sel, pending);
      if (!C.empty()) nearest = C[0].second;
    }
    return py::make_tuple(pending, changed, nearest);
  }

  void set_new(idx_t id, std::vector<idx_t> L) {
    auto it = staged0.find(id);
    changed += (int64_t)L.size() - (it == staged0.end() ? 0 : (int64_t)it->second.size());
    dirty.insert({0, id});
    staged0[id] = std::move(L);
  }

  py::tuple staged() const {
    py::dict d0;
    for (auto& kv : staged0) d0[py::int_(kv.first)] = py::cast(kv.second);
    py::list du;
    for (auto& kv : stagedu) du.append(py::make_tuple(kv.first.first, kv.first.second, kv.second));
    return py::make_tuple(d0, du);
  }

  // Changes since the last call, as (layer, node, added, removed) against the view's lists.
  py::list take_dirty() {
    py::list out;
    for (auto& k : dirty) {
      const std::vector<idx_t>& L = k.first == 0 ? staged0.at(k.second) : stagedu.at(k);
      std::vector<idx_t> base = k.second < v->n ? v->neighbors(k.second, k.first) : std::vector<idx_t>();
      std::unordered_set<idx_t> A(L.begin(), L.end()), B(base.begin(), base.end());
      std::vector<idx_t> added, removed;
      for (idx_t x : L)
        if (!B.count(x)) added.push_back(x);
      for (idx_t x : base)
        if (!A.count(x)) removed.push_back(x);
      out.append(py::make_tuple(k.first, k.second, added, removed));
    }
    dirty.clear();
    return out;
  }

  void load(py::dict d0, py::list du) {
    for (auto item : d0) {
      idx_t i = item.first.cast<idx_t>();
      auto L = item.second.cast<std::vector<idx_t>>();
      if (i >= v->n) set_new(i, L);
      else set(i, 0, L);
    }
    for (auto t : du) {
      auto tt = t.cast<std::tuple<int, idx_t, std::vector<idx_t>>>();
      set(std::get<1>(tt), std::get<0>(tt), std::get<2>(tt));
    }
    dirty.clear();
  }
};

// ---------------------------------------------------------------- bindings
PYBIND11_MODULE(_core, m) {
  py::class_<Map, std::shared_ptr<Map>>(m, "Map")
      .def_static("lowrank", &make_lowrank, py::arg("V"), py::arg("U"), py::arg("b"))
      .def_static("dense", &make_dense, py::arg("W"), py::arg("b"))
      .def("apply", [](const Map& mp, farr x) {
        if ((int)x.shape(1) != mp.din) throw std::invalid_argument("map input width");
        farr y({x.shape(0), (py::ssize_t)mp.dout});
        for (py::ssize_t r = 0; r < x.shape(0); ++r) mp.apply(x.data() + r * mp.din, y.mutable_data() + r * mp.dout);
        return y;
      })
      .def_readonly("din", &Map::din)
      .def_readonly("dout", &Map::dout);

  py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph")
      .def(py::init<int, int, int, float, uint64_t>(), py::arg("dim"), py::arg("max_degree"), py::arg("ef_construction"),
           py::arg("alpha"), py::arg("seed"))
      .def("build", &Graph::build, py::arg("vectors"), py::arg("threads") = 1)
      .def("insert", &Graph::insert)
      .def("reinsert", &Graph::reinsert)
      .def("search", &Graph::search, py::arg("q"), py::arg("ef"))
      .def("with_vectors", &Graph::with_vectors)
      .def("vectors", &Graph::vectors)
      .def("neighbors", &Graph::neighbors, py::arg("i"), py::arg("layer") = 0)
      .def("layer0", &Graph::layer0)
      .def("upper", &Graph::upper)
      .def_static("from_adjacency", &Graph::from_adjacency)
      .def("save", &Graph::save, py::arg("path"), py::arg("with_vectors") = true)
      .def_static("load", &Graph::load)
      .def_readonly("n", &Graph::n)
      .def_readonly("dim", &Graph::dim)
      .def_readonly("max_degree", &Graph::R);

  py::class_<Segment, std::shared_ptr<Segment>>(m, "Segment")
      .def(py::init<int, bool, farr>(), py::arg("width"), py::arg("legacy"), py::arg("rows"))
      .def("write_rows", &Segment::write_rows)
      .def("array", &Segment::array)
      .def_readonly("width", &Segment::width)
      .def_readonly("legacy", &Segment::legacy)
      .def_readonly("rows", &Segment::rows);

  py::class_<Overlay, std::shared_ptr<Overlay>>(m, "Overlay")
      .def(py::init<idx_t, int, uint64_t>(), py::arg("capacity"), py::arg("dim"), py::arg("version"))
      .def("put", &Overlay::put)
      .def("get", &Overlay::get_py)
      .def("has", [](const Overlay& o, idx_t i) { return o.get(i) != nullptr; })
      .def("kill", &Overlay::kill)
      .def("is_dead", &Overlay::is_dead)
      .def("items", &Overlay::items)
      .def_property_readonly("count", [](const Overlay& o) { return o.count.load(); })
      .def_readonly("capacity", &Overlay::cap)
      .def_readonly("version", &Overlay::version);

  py::class_<View, std::shared_ptr<View>>(m, "View")
      .def_static("initial", &View::initial)
      .def("search", &View::search, py::arg("q"), py::arg("ef"))
      .def("annotate", &View::annotate)
      .def("rotate_segment", &View::rotate_segment)
      .def("routing_vectors", &View::routing_vectors)
      .def("with_segment", &View::with_segment)
      .def("with_flags", &View::with_flags)
      .def("next", &View::next)
      .def("indegrees", &View::indegrees)
      .def("in_neighbors", &View::in_neighbors)
      .def("neighbors", &View::neighbors, py::arg("i"), py::arg("layer") = 0)
      .def("upper_layers", &View::upper_layers)
      .def("flags", [](const View& v) { return py::array(py::cast(*v.flags)); })
      .def("segment", [](const View& v, int s) { return v.segs[s]; })
      .def_property_readonly("live", [](const View& v) { return v.live; })
      .def_property_readonly("frozen", [](const View& v) { return v.frozen; })
      .def_readonly("epoch", &View::epoch)
      .def_readonly("n", &View::n)
      .def_readonly("dim", &View::dim)
      .def_readonly("seg_size", &View::seg_size)
      .def_property_readonly("segments", [](const View& v) { return v.segs.size(); });

  py::class_<Pin>(m, "Pin")
      .def("search", [](const Pin& p, farr q, int ef) { return p.view().search(q, ef); })
      .def("annotate", [](const Pin& p, farr q, iarr ids, farr d) { return p.view().annotate(q, ids, d); })
      .def("native_in_view", [](const Pin& p, iarr ids) {
        u8arr out(ids.size());
        for (py::ssize_t j = 0; j < ids.size(); ++j) out.mutable_data()[j] = ((*p.view().flags)[ids.data()[j]] & F_NATIVE) ? 1 : 0;
        return out;
      })
      .def_property_readonly("epoch", [](const Pin& p) { return p.view().epoch; })
      .def("release", &Pin::done)
      .def("__enter__", [](Pin& p) -> Pin& { return p; }, py::return_value_policy::reference)
      .def("__exit__", [](Pin& p, py::object, py::object, py::object) { p.done(); });

  py::class_<Holder>(m, "Holder")
      .def(py::init<>())
      .def("publish", [](Holder& h, std::shared_ptr<View> v) { h.publish(v); })
      .def("pin", [](Holder& h) { return std::make_unique<Pin>(h); }, py::keep_alive<0, 1>())
      .def("current", [](Holder& h) { return std::const_pointer_cast<View>(h.current()); });

  py::class_<Auditor>(m, "Auditor")
      .def(py::init([](std::shared_ptr<View> v, farr g, float a, int limit) { return std::make_unique<Auditor>(v, g, a, limit); }),
           py::arg("view"), py::arg("gamma"), py::arg("alpha"), py::arg("pool_limit"))
      .def("audit", &Auditor::audit)
      .def("link", &Auditor::link)
      .def("staged", &Auditor::staged)
      .def("take_dirty", &Auditor::take_dirty)
      .def("load", &Auditor::load)
      .def_readonly("changed", &Auditor::changed);

  m.def("certify", &certify, py::arg("d"), py::arg("exact"), py::arg("w"), py::arg("ids"), py::arg("k"));
  m.def("ambiguous", &ambiguous, py::arg("d"), py::arg("exact"), py::arg("w"), py::arg("ids"), py::arg("k"));
}
