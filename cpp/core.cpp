#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using Vec = std::vector<float>;
using Id = uint32_t;
using Pair = std::pair<double, Id>;
constexpr double INF = std::numeric_limits<double>::infinity();

double distance(const Vec& a, const Vec& b) {
    if (a.size() != b.size() || a.empty()) throw std::invalid_argument("distance dimensions");
    double s = 0;
    for (size_t i=0; i<a.size(); ++i) { double t=double(a[i])-b[i]; s+=t*t; }
    return std::sqrt(s);
}
void valid(const Vec& v) {
    if(v.empty()) throw std::invalid_argument("empty vector");
    for(auto x:v) if(!std::isfinite(x)) throw std::invalid_argument("nonfinite vector");
}
struct Node {
    Vec x;
    uint16_t region=0;
    uint8_t state=0; // legacy=0 bridged=1 resolving=2 native=3
    uint64_t fence=0;
    std::vector<std::vector<Id>> edges;
};
struct Candidate { Id id; double d; uint16_t region; uint8_t state; };

class Graph {
    mutable std::shared_mutex mu;
    std::vector<Node> nodes;
    std::vector<std::vector<std::set<Id>>> reverse;
    std::vector<std::vector<Vec>> maps;
    std::vector<Vec> biases;
    std::vector<double> eps, gamma;
    int M, efc, entry=-1, top=-1;
    double alpha;
    std::mt19937_64 rng;
    uint64_t generation=0;

    Vec vector_unlocked(Id i) const {
        const auto& n=nodes.at(i);
        if(n.state!=0 || maps.empty()) return n.x;
        const auto& w=maps.at(n.region); Vec out=biases.at(n.region);
        if(w.size()!=n.x.size()) throw std::runtime_error("bridge input width");
        for(size_t a=0;a<w.size();++a)
            for(size_t b=0;b<out.size();++b) out[b]+=n.x[a]*w[a][b];
        return out;
    }
    double dnode(Id a, Id b) const { return distance(vector_unlocked(a),vector_unlocked(b)); }
    double residual(Id i) const {
        return nodes[i].state==3 ? 0 : (gamma.empty()?0:gamma.at(nodes[i].region));
    }
    std::vector<Pair> layer(const Vec& q, Id start, int l, int ef,
                            std::set<Id>* scored=nullptr) const {
        std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>> todo;
        std::priority_queue<Pair> best;
        std::unordered_set<Id> seen;
        auto score=[&](Id i){ if(scored)scored->insert(i); return Pair(distance(q,vector_unlocked(i)),i); };
        auto p=score(start); todo.push(p); best.push(p); seen.insert(start);
        while(!todo.empty()) {
            auto cur=todo.top();todo.pop();
            if(best.size()>=size_t(ef) && cur>best.top()) break;
            if(l>=int(nodes[cur.second].edges.size())) continue;
            for(Id j:nodes[cur.second].edges[l]) if(seen.insert(j).second) {
                auto v=score(j);
                if(best.size()<size_t(ef) || v<best.top()) {
                    todo.push(v);best.push(v);
                    if(best.size()>size_t(ef))best.pop();
                }
            }
        }
        std::vector<Pair> out;
        while(!best.empty()){out.push_back(best.top());best.pop();}
        std::sort(out.begin(),out.end()); return out;
    }
    std::vector<Id> prune(Id p, std::vector<Id> pool,
                          std::vector<std::pair<Id,Id>>* ambiguous=nullptr) const {
        std::sort(pool.begin(),pool.end());pool.erase(std::unique(pool.begin(),pool.end()),pool.end());
        pool.erase(std::remove(pool.begin(),pool.end(),p),pool.end());
        std::vector<Pair> order;
        for(Id i:pool)order.emplace_back(dnode(p,i),i);
        std::sort(order.begin(),order.end()); std::vector<Id> chosen;
        for(auto [dp,c]:order) {
            bool remove=false;
            for(Id j:chosen) {
                double dc=dnode(j,c), rc=residual(j)+residual(c), rp=residual(p)+residual(c);
                if(ambiguous && rc+rp>0 && alpha*std::max(0.0,dc-rc)<=dp+rp &&
                   alpha*(dc+rc)>=std::max(0.0,dp-rp)) ambiguous->emplace_back(j,c);
                // DiskANN robust prune: occlude c if alpha*d(j,c) <= d(p,c).
                if(alpha*dc<=dp){remove=true;break;}
            }
            if(!remove) {chosen.push_back(c);if(chosen.size()>=size_t(M))break;}
        }
        return chosen;
    }
    void replace(Id p,int l,std::vector<Id> links) {
        for(Id j:nodes[p].edges[l])reverse[j][l].erase(p);
        nodes[p].edges[l]=std::move(links);
        for(Id j:nodes[p].edges[l])reverse[j][l].insert(p);
    }
    void rebuild_reverse() {
        reverse.clear();reverse.resize(nodes.size());
        for(size_t i=0;i<nodes.size();++i)reverse[i].resize(nodes[i].edges.size());
        for(Id i=0;i<nodes.size();++i)for(size_t l=0;l<nodes[i].edges.size();++l)
            for(Id j:nodes[i].edges[l]) {
                if(j>=nodes.size() || l>=nodes[j].edges.size())throw std::runtime_error("invalid graph edge");
                reverse[j][l].insert(i);
            }
    }
public:
    Graph(int m=32,int ec=200,double ap=1.2,uint64_t seed=42):M(m),efc(ec),alpha(ap),rng(seed) {
        if(m<2||ec<m||ap<1)throw std::invalid_argument("graph parameters");
    }
    void build(const std::vector<Vec>& data,const std::vector<uint16_t>& regions) {
        std::unique_lock lock(mu);
        if(data.size()!=regions.size())throw std::invalid_argument("build requires regions");
        size_t offset=nodes.size();
        for(Id local=0;local<data.size();++local) {
            Id id=Id(offset+local);
            valid(data[local]);if(id && data[local].size()!=nodes[0].x.size())throw std::invalid_argument("build widths");
            double u=std::generate_canonical<double,53>(rng);
            int level=std::min(32,int(-std::log(std::max(u,1e-15))/std::log(double(M))));
            Node n; n.x=data[local];n.region=regions[local];n.edges.resize(level+1);
            nodes.push_back(n);reverse.emplace_back(level+1);
            if(entry<0){entry=id;top=level;continue;}
            Id ep=entry;
            for(int l=top;l>level;--l)ep=layer(data[local],ep,l,1)[0].second;
            for(int l=std::min(level,top);l>=0;--l) {
                auto found=layer(data[local],ep,l,efc);std::vector<Id> pool;
                for(auto v:found)pool.push_back(v.second);
                auto links=prune(id,pool);replace(id,l,links);
                for(Id j:links){auto v=nodes[j].edges[l];v.push_back(id);replace(j,l,prune(j,v));}
                if(!found.empty())ep=found[0].second;
            }
            if(level>top){entry=id;top=level;}
        }
        ++generation;
    }
    void bridges(std::vector<std::vector<Vec>> w,std::vector<Vec> b,
                 std::vector<double> e,std::vector<double> g) {
        std::unique_lock lock(mu);
        if(w.size()!=b.size()||w.size()!=e.size()||w.size()!=g.size())throw std::invalid_argument("region arrays");
        for(size_t r=0;r<w.size();++r){valid(b[r]);for(auto& row:w[r]){valid(row);if(row.size()!=b[r].size())throw std::invalid_argument("bridge shape");}
            if(std::isnan(e[r])||std::isnan(g[r])||e[r]<0||g[r]<0)throw std::invalid_argument("radius");}
        maps=std::move(w);biases=std::move(b);eps=std::move(e);gamma=std::move(g);++generation;
    }
    void update(Id i,Vec x,uint8_t state,uint64_t fence) {
        valid(x);if(state>3)throw std::invalid_argument("state");
        std::unique_lock lock(mu);auto& n=nodes.at(i);
        if(fence<n.fence)throw std::runtime_error("stale fence");
        n.x=std::move(x);n.state=state;n.fence=fence;++generation;
    }
    bool publish(const std::vector<Id>& ids,const std::vector<Vec>& x,
                 const std::vector<uint8_t>& states,const std::vector<uint64_t>& fences,
                 py::function durable,py::function reclaim) {
        // The manifest adapter (RocksDB in this artifact) owns file/WAL sync.
        // Core readers cannot observe a publication until the durable callback
        // succeeds. Reclamation follows both durable and in-memory publication.
        if(ids.size()!=x.size()||ids.size()!=states.size()||ids.size()!=fences.size())
            throw std::invalid_argument("publication lengths");
        {
            std::unique_lock lock(mu);
            for(size_t a=0;a<ids.size();++a) {
                valid(x[a]);const auto& n=nodes.at(ids[a]);
                if(states[a]>3)throw std::runtime_error("invalid publication state");
                if(fences[a]<n.fence)return false;
            }
            if(!durable().cast<bool>())return false;
            for(size_t a=0;a<ids.size();++a) {
                auto& n=nodes[ids[a]];n.x=x[a];n.state=states[a];n.fence=fences[a];
            }
            ++generation;
        }
        reclaim();return true;
    }
    std::vector<Candidate> search(const Vec& q,int ef) const {
        valid(q);if(ef<1)throw std::invalid_argument("ef");
        std::shared_lock lock(mu);if(entry<0)return {};
        std::set<Id> scored;Id ep=entry;
        for(int l=top;l>0;--l)ep=layer(q,ep,l,1,&scored)[0].second;
        layer(q,ep,0,ef,&scored);
        std::vector<Candidate> out;
        for(Id i:scored)out.push_back({i,distance(q,vector_unlocked(i)),nodes[i].region,nodes[i].state});
        std::sort(out.begin(),out.end(),[](auto a,auto b){return Pair(a.d,a.id)<Pair(b.d,b.id);});return out;
    }
    std::vector<Candidate> rescore(const Vec& q,const std::vector<Id>& ids) const {
        std::shared_lock lock(mu);std::vector<Candidate> out;
        for(Id i:ids)out.push_back({i,distance(q,vector_unlocked(i)),nodes.at(i).region,nodes.at(i).state});
        std::sort(out.begin(),out.end(),[](auto a,auto b){return Pair(a.d,a.id)<Pair(b.d,b.id);});return out;
    }
    py::dict repair(Id p) {
        std::unique_lock lock(mu);nodes.at(p);size_t changed=0;
        std::vector<std::pair<Id,Id>> queued;
        for(int l=0;l<int(nodes[p].edges.size());++l) {
            std::set<Id> pool(nodes[p].edges[l].begin(),nodes[p].edges[l].end());
            pool.insert(reverse[p][l].begin(),reverse[p][l].end());auto first=pool;
            for(Id j:first){pool.insert(nodes[j].edges[l].begin(),nodes[j].edges[l].end());pool.insert(reverse[j][l].begin(),reverse[j][l].end());}
            auto old=nodes[p].edges[l];auto links=prune(p,{pool.begin(),pool.end()},&queued);
            for(Id x:old)if(std::find(links.begin(),links.end(),x)==links.end())++changed;
            for(Id x:links)if(std::find(old.begin(),old.end(),x)==old.end())++changed;
            replace(p,l,links);
        }
        if(changed)++generation;
        py::dict d;d["rewritten_endpoints"]=changed;d["ambiguous"]=queued;return d;
    }
    std::vector<size_t> indegrees() const {
        std::shared_lock lock(mu);std::vector<size_t> out;
        for(auto& n:reverse){size_t k=0;for(auto& l:n)k+=l.size();out.push_back(k);}return out;
    }
    bool check_reverse() const {
        std::shared_lock lock(mu);
        for(Id i=0;i<nodes.size();++i)for(size_t l=0;l<nodes[i].edges.size();++l) {
            for(Id j:nodes[i].edges[l])if(!reverse[j][l].count(i))return false;
            for(Id j:reverse[i][l])if(std::find(nodes[j].edges[l].begin(),nodes[j].edges[l].end(),i)==nodes[j].edges[l].end())return false;
        }return true;
    }
    py::dict topology() const {
        std::shared_lock lock(mu);std::vector<std::vector<std::vector<Id>>> all;
        for(auto& n:nodes)all.push_back(n.edges);
        py::dict d;d["edges"]=all;d["entry"]=entry;d["top"]=top;d["generation"]=generation;return d;
    }
    void restore(const std::vector<Vec>& x,const std::vector<uint16_t>& r,
                 const std::vector<uint8_t>& s,const std::vector<uint64_t>& f,py::dict t) {
        std::unique_lock lock(mu);
        auto e=t["edges"].cast<std::vector<std::vector<std::vector<Id>>>>();
        if(x.size()!=r.size()||x.size()!=s.size()||x.size()!=f.size()||x.size()!=e.size())throw std::invalid_argument("restore lengths");
        nodes.clear();
        for(size_t i=0;i<x.size();++i){valid(x[i]);nodes.push_back({x[i],r[i],s[i],f[i],e[i]});}
        entry=t["entry"].cast<int>();top=t["top"].cast<int>();generation=t["generation"].cast<uint64_t>();
        rebuild_reverse();
    }
    Vec vector(Id i) const {std::shared_lock lock(mu);return vector_unlocked(i);}
    uint64_t revision() const {std::shared_lock lock(mu);return generation;}
    void retire() {
        std::unique_lock lock(mu);for(auto& n:nodes)if(n.state!=3)throw std::runtime_error("incomplete migration");
        maps.clear();biases.clear();eps.clear();gamma.clear();++generation;
    }
};

std::pair<double,double> interval(const Candidate& c,double radius) {
    if(c.state==3)return {c.d,c.d};
    if(std::isnan(radius)||radius<0)throw std::invalid_argument("radius");
    return {std::max(0.0,c.d-radius),c.d+radius};
}
std::vector<Id> ambiguity(const std::vector<Candidate>& c,int k,const std::vector<double>& e) {
    if(k<1)throw std::invalid_argument("k");if(c.empty())return {};
    auto boundary=interval(c[std::min(size_t(k),c.size())-1],e.at(c[std::min(size_t(k),c.size())-1].region));
    std::vector<std::pair<double,Id>> out;
    for(auto x:c)if(x.state!=3) {
        auto a=interval(x,e.at(x.region));double lo=std::max(a.first,boundary.first),hi=std::min(a.second,boundary.second);
        if(lo<=hi)out.emplace_back(hi-lo,x.id);
    }
    std::sort(out.begin(),out.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
    std::vector<Id> ids;for(auto x:out)ids.push_back(x.second);return ids;
}
py::dict certificate(const std::vector<Candidate>& c,int k,double radius) {
    if(k<1)throw std::invalid_argument("k");
    size_t n=std::min(size_t(k),c.size()),m=0;double competitor=INF;
    for(size_t i=n;i<c.size();++i)competitor=std::min(competitor,interval(c[i],radius).first);
    for(size_t i=0;i<n;++i)if(n==c.size() || interval(c[i],radius).second<competitor)++m;
    py::dict d;d["m"]=m;d["bound"]=double(m)/k;d["candidate_count"]=c.size();d["insufficient_candidates"]=c.size()<size_t(k);
    d["competitor_min_lower"]=competitor;return d;
}
PYBIND11_MODULE(_core,m) {
    m.def("distance",&distance);m.def("interval",&interval);m.def("ambiguity",&ambiguity);m.def("certificate",&certificate);
    py::class_<Candidate>(m,"Candidate").def(py::init<Id,double,uint16_t,uint8_t>())
      .def_readonly("id",&Candidate::id).def_readonly("d",&Candidate::d).def_readonly("region",&Candidate::region).def_readonly("state",&Candidate::state);
    py::class_<Graph>(m,"Graph")
      .def(py::init<int,int,double,uint64_t>(),py::arg("M")=32,py::arg("efConstruction")=200,py::arg("alpha")=1.2,py::arg("seed")=42)
      .def("build",&Graph::build).def("append",&Graph::build).def("bridges",&Graph::bridges).def("update",&Graph::update)
      .def("publish",&Graph::publish)
      .def("search",&Graph::search).def("rescore",&Graph::rescore).def("repair",&Graph::repair)
      .def("indegrees",&Graph::indegrees).def("check_reverse",&Graph::check_reverse)
      .def("topology",&Graph::topology).def("restore",&Graph::restore).def("vector",&Graph::vector)
      .def("revision",&Graph::revision).def("retire",&Graph::retire);
}
