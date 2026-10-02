"""Durable dollar bucket. Reservations gate scheduled work; overruns create debt."""
import time
import uuid
import math


class Budget:
    def __init__(self, store, capacity, dollars_per_hour, price_per_hour):
        self.prices = dict(price_per_hour) if isinstance(price_per_hour, dict) else {'default': float(price_per_hour)}
        if any(not math.isfinite(v) or v < 0 for v in (capacity, dollars_per_hour, *self.prices.values())):
            raise ValueError('explicit nonnegative dollar prices/rates required')
        self.store, self.price = store, self.price_for('scheduled')
        if not store.get('budget'):
            store.transaction({'budget': dict(capacity=capacity, tokens=capacity, rate=dollars_per_hour / 3600,
                                              updated=time.time(), spent=0., paused=False, prices=self.prices), 'ledger': {}})
        elif store.get('budget')['prices'] != self.prices:
            raise ValueError('price configuration differs from durable budget; resume with recorded prices')

    def price_for(self, category):
        if category not in self.prices and 'default' not in self.prices:
            raise ValueError('missing explicit price for ' + category)
        return self.prices.get(category, self.prices.get('default'))

    def _refill(self, b, now):
        b['tokens'] = min(b['capacity'], b['tokens'] + max(0, now - b['updated']) * b['rate'])
        b['updated'] = now

    def reserve(self, dollars, job, now=None, metadata=None):
        if not math.isfinite(dollars) or dollars < 0:
            raise ValueError('negative reservation')
        now = time.time() if now is None else now
        with self.store.lock:
            b, ledger = self.store.get('budget'), self.store.get('ledger', {})
            if job in ledger:
                return False # no duplicate debit or execution grant
            self._refill(b, now)
            if b['paused'] or b['tokens'] < dollars:
                self.store.transaction({'budget': b})
                return False
            b['tokens'] -= dollars
            ledger[job] = dict(reserved=dollars, reconciled=False, actual=None, metadata=metadata, dispatched=False)
            self.store.transaction({'budget': b, 'ledger': ledger})
            return True

    def reconcile(self, job, dollars):
        if not math.isfinite(dollars) or dollars < 0:
            raise ValueError('negative actual cost')
        with self.store.lock:
            b, ledger = self.store.get('budget'), self.store.get('ledger')
            record = ledger[job]
            if record['reconciled']:
                return False
            b['tokens'] += record['reserved'] - dollars
            b['tokens'] = min(b['capacity'], b['tokens'])
            b['spent'] += dollars
            record.update(reconciled=True, actual=dollars)
            self.store.transaction({'budget': b, 'ledger': ledger})
            return True

    def account(self, category, seconds, count, device, job=None):
        if not math.isfinite(seconds) or seconds < 0 or count < 0:
            raise ValueError('invalid measured work')
        dollars = seconds / 3600 * self.price_for(category)
        with self.store.lock:
            work = self.store.get('work', [])
            work.append(dict(category=category, seconds=seconds, count=count, device=device, dollars=dollars,
                             gpu_busy_seconds=seconds if device.startswith('cuda') else 0, job=job))
            self.store.transaction({'work': work})
        # Non-scheduled work is observable but not gated by this bucket. Expedited
        # jobs need their own explicitly priced bucket if an operator wants a cap.
        return dollars

    def pause(self, value=True):
        with self.store.lock:
            b = self.store.get('budget')
            b['paused'] = value
            self.store.transaction({'budget': b})


class Scheduler:
    def __init__(self, index, budget, queue, estimated_seconds_per_item=1):
        self.index, self.budget, self.queue = index, budget, queue
        self.estimate = estimated_seconds_per_item

    def schedule(self, batch_size=32, max_items=None, origin='scheduled'):
        if origin not in ('scheduled', 'expedited'):
            raise ValueError('background origin must be scheduled or expedited')
        self.recover_pending()
        index = self.index
        with index.lock:
            nodes = index.store.nodes()
            heat = [index.store.get(f'heat/{i}', 0) for i in range(len(nodes))]
            degree = index.graph.indegrees()
            radii = index.store.get('bridges')
            groups = {}
            for i, n in enumerate(nodes):
                if n['state'] not in ('legacy', 'bridged'):
                    continue
                e = radii[n['region']]['epsilon'] if radii else 0
                # Avoid 0*infinity NaN; zero heat or in-degree means zero utility.
                u = heat[i] * e * degree[i] if heat[i] and degree[i] else 0
                groups.setdefault(n['region'], []).append((u, i))
            ordered = sorted(groups, key=lambda r: (-max(x[0] for x in groups[r]), r))
        jobs = 0
        submitted = 0
        for r in ordered:
            ids = [i for _, i in sorted(groups[r], key=lambda x: (-x[0], x[1]))]
            for start in range(0, len(ids), batch_size):
                batch = ids[start:start + batch_size]
                if max_items is not None:
                    batch = batch[:max(0, max_items - submitted)]
                    if not batch:
                        return jobs
                job_id = uuid.uuid4().hex
                if not self.budget.reserve(len(batch) * self.estimate * self.budget.price_for(origin) / 3600, job_id,
                                           metadata=dict(ids=batch, region=r, origin=origin)):
                    return jobs
                # Atomic durable queue message and claims under index host lock.
                with index.lock:
                    claims = [c for i in batch if (c := index.claim(i, job_id))]
                    if claims:
                        self.queue.send(dict(id=job_id, claims=claims, origin=origin, region=r))
                        ledger = index.store.get('ledger')
                        ledger[job_id]['dispatched'] = True
                        index.store.transaction({'ledger': ledger})
                        jobs += 1
                        submitted += len(claims)
                    else:
                        self.budget.reconcile(job_id, 0)
        return jobs

    def recover_pending(self):
        """Durable outbox metadata recovers reserve/claim/send crash gaps."""
        index = self.index
        with index.lock:
            ledger = index.store.get('ledger', {})
            for job, record in list(ledger.items()):
                meta = record.get('metadata')
                if not meta or record['reconciled'] or record.get('dispatched'):
                    continue
                claims, busy = [], False
                for i in meta['ids']:
                    n = index.store.node(i)
                    if n['state'] == 'native':
                        continue
                    if n['state'] == 'resolving' and n['owner'] == job and n['expires'] > time.time():
                        claims.append(dict(id=i, owner=job, fence=n['fence'], expires=n['expires']))
                    else:
                        c = index.claim(i, job, 120)
                        if c:
                            claims.append(c)
                        else:
                            busy = True
                if busy:
                    continue
                if claims:
                    self.queue.send(dict(id=job, claims=claims, region=meta['region'], origin=meta['origin']))
                    current = index.store.get('ledger')
                    current[job]['dispatched'] = True
                    index.store.transaction({'ledger': current})
                else:
                    dollars = sum(w['dollars'] for w in index.store.get('work', []) if w.get('job') == job)
                    self.budget.reconcile(job, dollars)
