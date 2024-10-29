# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Graph algorithm generators.

Currently implements the following:
- Depth-first search (Moore, 1959)
- Breadth-first search (Moore, 1959)
- Topological sorting (Knuth, 1973)
- Articulation points
- Bridges
- Kosaraju's strongly-connected components (Aho et al., 1974)
- Kruskal's minimum spanning tree (Kruskal, 1956)
- Prim's minimum spanning tree (Prim, 1957)
- Bellman-Ford's single-source shortest path (Bellman, 1958)
- Dijkstra's single-source shortest path (Dijkstra, 1959)
- DAG shortest path
- Floyd-Warshall's all-pairs shortest paths (Floyd, 1962)
- Edmonds-Karp bipartite matching (Edmund & Karp, 1972)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name

import torch

from algo_reasoning.src.data import AlgorithmicData
from algo_reasoning.src.specs import OutputClass
from algo_reasoning.src.probing import graph, array_cat, mask_one

def dfs(A, nb_nodes, *args, **kwargs):
    """Depth-first search (Moore, 1959)."""

    data = AlgorithmicData(algorithm="dfs", *args, **kwargs)

    data.set_inputs({
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    color = torch.zeros(A.size(0), dtype=torch.int32)
    pi = torch.arange(A.size(0))
    d = torch.zeros(A.size(0))
    f = torch.zeros(A.size(0))
    s_prev = torch.arange(A.size(0))
    time = 0

    for s in range(A.size(0)):
        if color[s].item() == 0:
            s_last = s
            u = s
            v = s

            data.increase_hints({
                'pi_h': pi.clone(),
                'color': array_cat(color, 3),
                'd': d.clone(),
                'f': f.clone(),
                's_prev': s_prev.clone(),
                's': mask_one(s, A.size(0)),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                's_last': mask_one(s_last, A.size(0)),
                'time': torch.tensor(time)
            })
            while True:
                if color[u].item() == 0 or d[u].item() == 0.0:
                    time += 0.01
                    d[u] = time
                    color[u] = 1

                    data.increase_hints({
                        'pi_h': pi.clone(),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time)
                    })

                for v in range(A.size(0)):
                    if A[u, v].item() != 0:
                        if color[v].item() == 0:
                            pi[v] = u
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v

                            data.increase_hints({
                                'pi_h': pi.clone(),
                                'color': array_cat(color, 3),
                                'd': d.clone(),
                                'f': f.clone(),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0)),
                                'time': torch.tensor(time)
                            })
                            break

                if s_last == u:
                    color[u] = 2
                    time += 0.01
                    f[u] = time

                    data.increase_hints({
                        'pi_h': pi.clone(),
                        'color': array_cat(color, 3),
                        'd':d.clone(),
                        'f':f.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time)
                    })

                    if s_prev[u].item() == u:
                        assert s_prev[s_last].item() == s_last
                        break
                    pr = s_prev[s_last].item()
                    s_prev[s_last] = s_last
                    s_last = pr

                u = s_last

    data.set_outputs({
        'pi': pi.clone()
    })

    return data

def bfs(A, s, nb_nodes, *args, **kwargs):
    """Breadth-first search (Moore, 1959)."""
    
    data = AlgorithmicData(algorithm="bfs", *args, **kwargs)

    data.set_inputs({
        's': mask_one(s, A.size(0)),
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    reach = torch.zeros(A.size(0))
    pi = torch.arange(A.size(0))
    reach[s] = 1
    while True:
        prev_reach = reach.clone()

        data.increase_hints({
            'reach_h': prev_reach.clone(),
            'pi_h': pi.clone()
        })

        for i in range(A.size(0)):
            for j in range(A.size(0)):
                if A[i, j].item() > 0 and prev_reach[i].item() == 1:
                    if pi[j] == j and j != s:
                        pi[j] = i
                    reach[j] = 1
        if torch.all(reach == prev_reach):
            break

    data.set_outputs({
        'pi': pi.clone()
    })

    return data

def topological_sort(A, nb_nodes, *args, **kwargs):
    """Topological sorting (Knuth, 1973)."""

    data = AlgorithmicData(algorithm="topological_sort", *args, **kwargs)

    data.set_inputs({
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    color = torch.zeros(A.size(0), dtype=torch.int32)
    topo = torch.arange(A.size(0))
    s_prev = torch.arange(A.size(0))
    topo_head = 0
    for s in range(A.size(0)):
        if color[s] == 0:
            s_last = s
            u = s
            v = s

            data.increase_hints({
                'topo_h': topo.clone(),
                'topo_head_h': mask_one(topo_head, A.size(0)),
                'color': array_cat(color, 3),
                's_prev': s_prev.clone(),
                's': mask_one(s, A.size(0)),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                's_last': mask_one(s_last, A.size(0))
            })

            while True:
                if color[u] == 0:
                    color[u] = 1

                    data.increase_hints({
                        'topo_h': topo.clone(),
                        'topo_head_h': mask_one(topo_head, A.size(0)),
                        'color': array_cat(color, 3),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0))
                    })

                for v in range(A.size(0)):
                    if A[u, v].item() != 0:
                        if color[v].item() == 0:
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v

                            data.increase_hints({
                                'topo_h': topo.clone(),
                                'topo_head_h': mask_one(topo_head, A.size(0)),
                                'color': array_cat(color, 3),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0))
                            })
                            break

                if s_last == u:
                    color[u] = 2

                    if color[topo_head].item() == 2:
                        topo[u] = topo_head
                    topo_head = u

                    data.increase_hints({
                        'topo_h': topo.clone(),
                        'topo_head_h': mask_one(topo_head, A.size(0)),
                        'color': array_cat(color, 3),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0))
                    })

                    if s_prev[u] == u:
                        assert s_prev[s_last] == s_last
                        break
                    pr = s_prev[s_last].item()
                    s_prev[s_last] = s_last
                    s_last = pr

                u = s_last

    data.set_outputs({
        'topo': topo.clone(),
        'topo_head': mask_one(topo_head, A.size(0))
    })

    return data

def articulation_points(A, nb_nodes, *args, **kwargs):
    """Articulation points."""

    data = AlgorithmicData(algorithm="articulation_points", *args, **kwargs)

    data.set_inputs({
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)


    color = torch.zeros(A.size(0), dtype=torch.int32)
    pi = torch.arange(A.size(0))
    d = torch.zeros(A.size(0))
    f = torch.zeros(A.size(0))
    s_prev = torch.arange(A.size(0))
    time = 0

    low = torch.zeros(A.size(0))
    child_cnt = torch.zeros(A.size(0))
    is_cut = torch.zeros(A.size(0))

    for s in range(A.size(0)):
        if color[s].item() == 0:
            s_last = s
            u = s
            v = s

            data.increase_hints({
                'is_cut_h': is_cut.clone(),
                'pi_h': pi.clone(),
                'color': array_cat(color, 3),
                'd': d.clone(),
                'f': f.clone(),
                'low': low.clone(),
                'child_cnt': child_cnt.clone(),
                's_prev': s_prev.clone(),
                's': mask_one(s, A.size(0)),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                's_last': mask_one(s_last, A.size(0)),
                'time': torch.tensor(time)
            })

            while True:
                if color[u].item() == 0 or d[u].item() == 0.0:
                    time += 0.01
                    d[u] = time
                    low[u] = time
                    color[u] = 1

                    data.increase_hints({
                        'is_cut_h': is_cut.clone(),
                        'pi_h': pi.clone(),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        'low': low.clone(),
                        'child_cnt': child_cnt.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time)
                    })

                for v in range(A.size(0)):
                    if A[u, v].item() != 0:
                        if color[v].item() == 0:
                            pi[v] = u
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v
                            child_cnt[u] += 0.01

                            data.increase_hints({
                                'is_cut_h': is_cut.clone(),
                                'pi_h': pi.clone(),
                                'color': array_cat(color, 3),
                                'd': d.clone(),
                                'f': f.clone(),
                                'low': low.clone(),
                                'child_cnt': child_cnt.clone(),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0)),
                                'time': torch.tensor(time)
                            })

                            break
                        elif v != pi[u].item():
                            low[u] = min(low[u].item(), d[v].item())

                            data.increase_hints({
                                'is_cut_h': is_cut.clone(),
                                'pi_h': pi.clone(),
                                'color': array_cat(color, 3),
                                'd': d.clone(),
                                'f': f.clone(),
                                'low': low.clone(),
                                'child_cnt': child_cnt.clone(),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0)),
                                'time': torch.tensor(time)
                            })

                if s_last == u:
                    color[u] = 2
                    time += 0.01
                    f[u] = time

                    for v in range(A.size(0)):
                        if pi[v] == u:
                            low[u] = min(low[u].item(), low[v].item())
                            if pi[u].item() != u and low[v].item() >= d[u].item():
                                is_cut[u] = 1
                    if pi[u].item() == u and child_cnt[u].item() > 0.01:
                        is_cut[u] = 1

                    data.increase_hints({
                        'is_cut_h': is_cut.clone(),
                        'pi_h': pi.clone(),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        'low': low.clone(),
                        'child_cnt': child_cnt.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time)
                    })

                    if s_prev[u].item() == u:
                        assert s_prev[s_last] == s_last
                        break
                    pr = s_prev[s_last].item()
                    s_prev[s_last] = s_last
                    s_last = pr

                u = s_last

    data.set_outputs({
        'is_cut': is_cut.clone()
    })
    
    return data

def bridges(A, nb_nodes, *args, **kwargs):
    """Bridges."""
    data = AlgorithmicData(algorithm="bridges", *args, **kwargs)

    adj = graph(A.clone())

    data.set_inputs({
        'A': A.clone(),
        'adj': adj
    }, nb_nodes)
    
    color = torch.zeros(A.size(0), dtype=torch.int32)
    pi = torch.arange(A.size(0))
    d = torch.zeros(A.size(0))
    f = torch.zeros(A.size(0))
    s_prev = torch.arange(A.size(0))
    time = 0

    low = torch.zeros(A.size(0))
    is_bridge = (torch.zeros((A.size(0), A.size(0))) + OutputClass.MASKED + adj)

    for s in range(A.size(0)):
        if color[s] == 0:
            s_last = s
            u = s
            v = s
            data.increase_hints({
					'is_bridge_h':  is_bridge.clone(),
					'pi_h': pi.clone(),
					'color': array_cat(color, 3),
					'd': d.clone(),
					'f': f.clone(),
					'low': low.clone(),
					's_prev': s_prev.clone(),
					's': mask_one(s, A.size(0)),
					'u': mask_one(u, A.size(0)),
					'v': mask_one(v, A.size(0)),
					's_last': mask_one(s_last, A.size(0)),
					'time': torch.tensor(time)
			})
            while True:
                if color[u].item() == 0 or d[u].item() == 0.0:
                    time += 0.01
                    d[u] = time
                    low[u] = time
                    color[u] = 1
                    data.increase_hints({
                        'is_bridge_h': is_bridge.clone(),
                        'pi_h': pi.clone(),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        'low': low.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time)
                    })
                
                for v in range(A.size(0)):
                    if A[u, v].item() != 0:
                        if color[v].item() == 0:
                            pi[v] = u
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v


                            data.increase_hints({
                                'is_bridge_h': is_bridge.clone(),
                                'pi_h': pi.clone(),
                                'color': array_cat(color, 3),
                                'd': d.clone(),
                                'f': f.clone(),
                                'low': low.clone(),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0)),
                                'time': torch.tensor(time)
                            })
                            break
                        elif v != pi[u]:
                            low[u] = min(low[u].item(), d[v].item())

                            data.increase_hints({
                                'is_bridge_h': is_bridge.clone(),
                                'pi_h': pi.clone(),
                                'color': array_cat(color, 3),
                                'd': d.clone(),
                                'f': f.clone(),
                                'low': low.clone(),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0)),
                                'time': torch.tensor(time)
                            })
                
                if s_last == u:
                    color[u] = 2
                    time += 0.01
                    f[u] = time
                    
                    for v in range(A.size(0)):
                        if pi[v] == u:
                            low[u] = min(low[u].item(), low[v].item())
                            if low[v] > d[u]:
                                is_bridge[u, v] = 1
                                is_bridge[v, u] = 1

                    data.increase_hints({
                        'is_bridge_h': is_bridge.clone(),
                        'pi_h': pi.clone(),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        'low': low.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time)
                    })
                    if s_prev[u] == u:
                        assert s_prev[s_last] == s_last
                        break

                    pr = s_prev[s_last].item()
                    s_prev[s_last] = s_last
                    s_last = pr
                
                u = s_last

    data.set_outputs({
        'is_bridge': is_bridge.clone()
    })

    return data

def strongly_connected_components(A, nb_nodes, *args, **kwargs):
    """Kosaraju's strongly-connected components (Aho et al., 1974)."""

    data = AlgorithmicData(algorithm="strongly_connected_components", *args, **kwargs)

    data.set_inputs({
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)


    scc_id = torch.arange(A.size(0))
    color = torch.zeros(A.size(0), dtype=torch.int32)
    d = torch.zeros(A.size(0))
    f = torch.zeros(A.size(0))
    s_prev = torch.arange(A.size(0))
    time = 0
    A_t = torch.transpose(A, 0, 1)

    for s in range(A.size(0)):
        if color[s].item() == 0:
            s_last = s
            u = s
            v = s

            data.increase_hints({
                'scc_id_h': scc_id.clone(),
                'A_t': graph(A_t.clone()),
                'color': array_cat(color, 3),
                'd': d.clone(),
                'f': f.clone(),
                's_prev': s_prev.clone(),
                's': mask_one(s, A.size(0)),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                's_last': mask_one(s_last, A.size(0)),
                'time': torch.tensor(time),
                'phase': torch.tensor(0)
            })

            while True:
                if color[u].item() == 0 or d[u].item() == 0.0:
                    time += 0.01
                    d[u] = time
                    color[u] = 1

                    data.increase_hints({
                        'scc_id_h': scc_id.clone(),
                        'A_t': graph(A_t.clone()),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time),
                        'phase': torch.tensor(0)
                    })

                for v in range(A.size(0)):
                    if A[u, v].item() != 0:
                        if color[v].item() == 0:
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v

                            data.increase_hints({
                                'scc_id_h': scc_id.clone(),
                                'A_t': graph(A_t.clone()),
                                'color': array_cat(color, 3),
                                'd': d.clone(),
                                'f': f.clone(),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0)),
                                'time': torch.tensor(time),
                                'phase': torch.tensor(0)
                            })

                            break

                if s_last == u:
                    color[u] = 2
                    time += 0.01
                    f[u] = time

                    data.increase_hints({
                        'scc_id_h': scc_id.clone(),
                        'A_t': graph(A_t.clone()),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time),
                        'phase': torch.tensor(0)
                    })

                    if s_prev[u] == u:
                        assert s_prev[s_last].item() == s_last
                        break
                    pr = s_prev[s_last].item()
                    s_prev[s_last] = s_last
                    s_last = pr

                u = s_last

    color = torch.zeros(A.size(0), dtype=torch.int32)
    s_prev = torch.arange(A.size(0))

    for s in torch.argsort(-f).tolist():
        if color[s].item() == 0:
            s_last = s
            u = s
            v = s

            data.increase_hints({
                'scc_id_h': scc_id.clone(),
                'A_t': graph(A_t.clone()),
                'color': array_cat(color, 3),
                'd': d.clone(),
                'f': f.clone(),
                's_prev': s_prev.clone(),
                's': mask_one(s, A.size(0)),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                's_last': mask_one(s_last, A.size(0)),
                'time': torch.tensor(time),
                'phase': torch.tensor(1)
            })
            while True:
                scc_id[u] = s
                if color[u].item() == 0 or d[u].item() == 0.0:
                    time += 0.01
                    d[u] = time
                    color[u] = 1

                    data.increase_hints({
                        'scc_id_h': scc_id.clone(),
                        'A_t': graph(A_t.clone()),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time),
                        'phase': torch.tensor(1)
                    })

                for v in range(A.size(0)):
                    if A_t[u, v].item() != 0:
                        if color[v].item() == 0:
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v

                            data.increase_hints({
                                'scc_id_h': scc_id.clone(),
                                'A_t': graph(A_t.clone()),
                                'color': array_cat(color, 3),
                                'd': d.clone(),
                                'f': f.clone(),
                                's_prev': s_prev.clone(),
                                's': mask_one(s, A.size(0)),
                                'u': mask_one(u, A.size(0)),
                                'v': mask_one(v, A.size(0)),
                                's_last': mask_one(s_last, A.size(0)),
                                'time': torch.tensor(time),
                                'phase': torch.tensor(1)
                            })

                            break

                if s_last == u:
                    color[u] = 2
                    time += 0.01
                    f[u] = time

                    data.increase_hints({
                        'scc_id_h': scc_id.clone(),
                        'A_t': graph(A_t.clone()),
                        'color': array_cat(color, 3),
                        'd': d.clone(),
                        'f': f.clone(),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'time': torch.tensor(time),
                        'phase': torch.tensor(1)
                    })

                    if s_prev[u] == u:
                        assert s_prev[s_last] == s_last
                        break
                    pr = s_prev[s_last].item()
                    s_prev[s_last] = s_last
                    s_last = pr

                u = s_last

    data.set_outputs({
        'scc_id': scc_id.clone()
    })

    return data

def mst_kruskal(A, nb_nodes, *args, **kwargs):
    """Kruskal's minimum spanning tree (Kruskal, 1956)."""

    data = AlgorithmicData(algorithm="mst_kruskal", *args, **kwargs)

    data.set_inputs({
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    pi = torch.arange(A.size(0))

    def mst_union(u, v, in_mst):
        root_u = u
        root_v = v

        mask_u = torch.zeros(in_mst.size(0))
        mask_v = torch.zeros(in_mst.size(0))

        mask_u[u] = 1
        mask_v[v] = 1

        data.increase_hints({
            'in_mst_h': in_mst.clone(),
            'pi': pi.clone(),
            'u': mask_one(u, A.size(0)),
            'v': mask_one(v, A.size(0)),
            'root_u': mask_one(root_u, A.size(0)),
            'root_v': mask_one(root_v, A.size(0)),
            'mask_u': mask_u.clone(),
            'mask_v': mask_v.clone(),
            'phase': mask_one(1, 3)
        })


        while pi[root_u].item() != root_u:
            root_u = pi[root_u].item()
            for i in range(mask_u.size(0)):
                if mask_u[i].item() == 1:
                    pi[i] = root_u
            mask_u[root_u] = 1

            data.increase_hints({
                'in_mst_h': in_mst.clone(),
                'pi': pi.clone(),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                'root_u': mask_one(root_u, A.size(0)),
                'root_v': mask_one(root_v, A.size(0)),
                'mask_u': mask_u.clone(),
                'mask_v': mask_v.clone(),
                'phase': mask_one(1, 3)
            })

        while pi[root_v].item() != root_v:
            root_v = pi[root_v].item()
            for i in range(mask_v.size(0)):
                if mask_v[i] == 1:
                    pi[i] = root_v
            mask_v[root_v] = 1

            data.increase_hints({
                'in_mst_h': in_mst.clone(),
                'pi': pi.clone(),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                'root_u': mask_one(root_u, A.size(0)),
                'root_v': mask_one(root_v, A.size(0)),
                'mask_u': mask_u.clone(),
                'mask_v': mask_v.clone(),
                'phase': mask_one(2, 3)
            })

        if root_u < root_v:
            in_mst[u, v] = 1
            in_mst[v, u] = 1
            pi[root_u] = root_v
        elif root_u > root_v:
            in_mst[u, v] = 1
            in_mst[v, u] = 1
            pi[root_v] = root_u
        
        data.increase_hints({
            'in_mst_h': in_mst.clone(),
            'pi': pi.clone(),
            'u': mask_one(u, A.size(0)),
            'v': mask_one(v, A.size(0)),
            'root_u': mask_one(root_u, A.size(0)),
            'root_v': mask_one(root_v, A.size(0)),
            'mask_u': mask_u.clone(),
            'mask_v': mask_v.clone(),
            'phase': mask_one(0, 3)
        })

    in_mst = torch.zeros((A.size(0), A.size(0)))

    # Prep to sort edge array
    lx = []
    ly = []
    wts = []
    for i in range(A.size(0)):
        for j in range(i + 1, A.size(0)):
            if A[i, j] > 0:
                lx.append(i)
                ly.append(j)
                wts.append(A[i, j].item())

    data.increase_hints({
        'in_mst_h': in_mst.clone(),
        'pi': pi.clone(),
        'u': mask_one(0, A.size(0)),
        'v': mask_one(0, A.size(0)),
        'root_u': mask_one(0, A.size(0)),
        'root_v': mask_one(0, A.size(0)),
        'mask_u': torch.zeros(A.size(0)),
        'mask_v': torch.zeros(A.size(0)),
        'phase': mask_one(0, 3)
    })

    for ind in torch.argsort(torch.tensor(wts)).tolist():
        u = lx[ind]
        v = ly[ind]
        mst_union(u, v, in_mst)

    data.set_outputs({
        'in_mst': in_mst.clone()
    })

    return data

def mst_prim(A, s, nb_nodes, *args, **kwargs):
    """Prim's minimum spanning tree (Prim, 1957)."""

    data = AlgorithmicData(algorithm="mst_prim", *args, **kwargs)

    data.set_inputs({
        's': mask_one(s, A.size(0)),
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    key = torch.zeros(A.size(0))
    mark = torch.zeros(A.size(0))
    in_queue = torch.zeros(A.size(0))
    pi = torch.arange(A.size(0))
    key[s] = 0
    in_queue[s] = 1

    data.increase_hints({
        'pi_h': pi.clone(),
        'key': key.clone(),
        'mark': mark.clone(),
        'in_queue': in_queue.clone(),
        'u': mask_one(s, A.size(0))
    })

    for _ in range(A.size(0)):
        u = torch.argsort(key + (1.0 - in_queue) * 1e9)[0].item()
        if in_queue[u].item() == 0:
            break
        mark[u] = 1
        in_queue[u] = 0
        for v in range(A.size(0)):
            if A[u, v].item() != 0:
                if mark[v].item() == 0 and (in_queue[v].item() == 0 or A[u, v].item() < key[v].item()):
                    pi[v] = u
                    key[v] = A[u, v].item()
                    in_queue[v] = 1

        data.increase_hints({
            'pi_h': pi.clone(),
            'key': key.clone(),
            'mark': mark.clone(),
            'in_queue': in_queue.clone(),
            'u': mask_one(u, A.size(0))
        })

    data.set_outputs({
        'pi': pi.clone()
    })

    return data

def bellman_ford(A, s, nb_nodes, *args, **kwargs):
    """Bellman-Ford's single-source shortest path (Bellman, 1958)."""

    data = AlgorithmicData(algorithm="bellman_ford", *args, **kwargs)

    data.set_inputs({
        's': mask_one(s, A.size(0)),
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    d = torch.zeros(A.size(0))
    pi = torch.arange(A.size(0))
    msk = torch.zeros(A.size(0))
    d[s] = 0
    msk[s] = 1

    while True:
        prev_d = d.clone()
        prev_msk = msk.clone()

        data.increase_hints({
            'pi_h': pi.clone(),
            'd': prev_d.clone(),
            'msk': prev_msk.clone()
        })

        for u in range(A.size(0)):
            for v in range(A.size(0)):
                if prev_msk[u].item() == 1 and A[u, v].item() != 0:
                    if msk[v].item() == 0 or prev_d[u].item() + A[u, v].item() < d[v].item():
                        d[v] = prev_d[u].item() + A[u, v].item()
                        pi[v] = u
                    msk[v] = 1

        if torch.all(d == prev_d):
            break

    data.set_outputs({
        'pi': pi.clone()
    })

    return data

def dijkstra(A, s, nb_nodes, *args, **kwargs):
    """Dijkstra's single-source shortest path (Dijkstra, 1959)."""

    data = AlgorithmicData(algorithm="dijkstra", *args, **kwargs)

    data.set_inputs({
        's': mask_one(s, A.size(0)),
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    d = torch.zeros(A.size(0))
    mark = torch.zeros(A.size(0))
    in_queue = torch.zeros(A.size(0))
    pi = torch.arange(A.size(0))
    d[s] = 0
    in_queue[s] = 1

    data.increase_hints({
        'pi_h': pi.clone(),
        'd': d.clone(),
        'mark': mark.clone(),
        'in_queue': in_queue.clone(),
        'u': mask_one(s, A.size(0))
    })

    for _ in range(A.size(0)):
        u = torch.argsort(d + (1.0 - in_queue) * 1e9)[0].item() 
        if in_queue[u].item() == 0:
            break
        mark[u] = 1
        in_queue[u] = 0
        for v in range(A.size(0)):
            if A[u, v].item() != 0:
                if mark[v].item() == 0 and (in_queue[v].item() == 0 or d[u].item() + A[u, v].item() < d[v].item()):
                    pi[v] = u
                    d[v] = d[u].item() + A[u, v].item()
                    in_queue[v] = 1

        data.increase_hints({
            'pi_h': pi.clone(),
            'd': d.clone(),
            'mark': mark.clone(),
            'in_queue': in_queue.clone(),
            'u': mask_one(u, A.size(0))
        })

    data.set_outputs({
        'pi': pi.clone()
    })

    return data

def dag_shortest_paths(A, s, nb_nodes, *args, **kwargs):
    """DAG shortest path."""

    data = AlgorithmicData(algorithm="dag_shortest_paths", *args, **kwargs)

    data.set_inputs({
        's': mask_one(s, A.size(0)),
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    pi = torch.arange(A.size(0))
    d = torch.zeros(A.size(0))
    mark = torch.zeros(A.size(0))
    color = torch.zeros(A.size(0), dtype=torch.int32)
    topo = torch.arange(A.size(0))
    s_prev = torch.arange(A.size(0))
    topo_head = 0
    s_last = s
    u = s
    v = s

    data.increase_hints({
        'pi_h': pi.clone(),
        'd': d.clone(),
        'mark': mark.clone(),
        'topo_h': topo.clone(),
        'topo_head_h': mask_one(topo_head, A.size(0)),
        'color': array_cat(color, 3),
        's_prev': s_prev.clone(),
        's': mask_one(s, A.size(0)),
        'u': mask_one(u, A.size(0)),
        'v': mask_one(v, A.size(0)),
        's_last': mask_one(s_last, A.size(0)),
        'phase': torch.tensor(0)
    })

    while True:
        if color[u].item() == 0:
            color[u] = 1
            data.increase_hints({
                'pi_h': pi.clone(),
                'd': d.clone(),
                'mark': mark.clone(),
                'topo_h': topo.clone(),
                'topo_head_h': mask_one(topo_head, A.size(0)),
                'color': array_cat(color, 3),
                's_prev': s_prev.clone(),
                's': mask_one(s, A.size(0)),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                's_last': mask_one(s_last, A.size(0)),
                'phase': torch.tensor(0)
            })

        for v in range(A.size(0)):
            if A[u, v].item() != 0:
                if color[v].item() == 0:
                    color[v] = 1
                    s_prev[v] = s_last
                    s_last = v

                    data.increase_hints({
                        'pi_h': pi.clone(),
                        'd': d.clone(),
                        'mark': mark.clone(),
                        'topo_h': topo.clone(),
                        'topo_head_h': mask_one(topo_head, A.size(0)),
                        'color': array_cat(color, 3),
                        's_prev': s_prev.clone(),
                        's': mask_one(s, A.size(0)),
                        'u': mask_one(u, A.size(0)),
                        'v': mask_one(v, A.size(0)),
                        's_last': mask_one(s_last, A.size(0)),
                        'phase': torch.tensor(0)
                    })
                    break

        if s_last == u:
            color[u] = 2

            if color[topo_head].item() == 2:
                topo[u] = topo_head
            topo_head = u

            data.increase_hints({
                'pi_h': pi.clone(),
                'd': d.clone(),
                'mark': mark.clone(),
                'topo_h': topo.clone(),
                'topo_head_h': mask_one(topo_head, A.size(0)),
                'color': array_cat(color, 3),
                's_prev': s_prev.clone(),
                's': mask_one(s, A.size(0)),
                'u': mask_one(u, A.size(0)),
                'v': mask_one(v, A.size(0)),
                's_last': mask_one(s_last, A.size(0)),
                'phase': torch.tensor(0)
            })

            if s_prev[u].item() == u:
                assert s_prev[s_last].item() == s_last
                break
            pr = s_prev[s_last].item()
            s_prev[s_last] = s_last
            s_last = pr

        u = s_last

    assert topo_head == s
    d[topo_head] = 0
    mark[topo_head] = 1

    while topo[topo_head] != topo_head:
        i = topo_head
        mark[topo_head] = 1

        data.increase_hints({
            'pi_h': pi.clone(),
            'd': d.clone(),
            'mark': mark.clone(),
            'topo_h': topo.clone(),
            'topo_head_h': mask_one(topo_head, A.size(0)),
            'color': array_cat(color, 3),
            's_prev': s_prev.clone(),
            's': mask_one(s, A.size(0)),
            'u': mask_one(u, A.size(0)),
            'v': mask_one(v, A.size(0)),
            's_last': mask_one(s_last, A.size(0)),
            'phase': torch.tensor(1)
        })

        for j in range(A.size(0)):
            if A[i, j].item() != 0.0:
                if mark[j].item() == 0 or d[i].item() + A[i, j].item() < d[j].item():
                    d[j] = d[i].item() + A[i, j].item()
                    pi[j] = i
                    mark[j] = 1

        topo_head = topo[topo_head].item()

    data.increase_hints({
        'pi_h': pi.clone(),
        'd': d.clone(),
        'mark': mark.clone(),
        'topo_h': topo.clone(),
        'topo_head_h': mask_one(topo_head, A.size(0)),
        'color': array_cat(color, 3),
        's_prev': s_prev.clone(),
        's': mask_one(s, A.size(0)),
        'u': mask_one(u, A.size(0)),
        'v': mask_one(v, A.size(0)),
        's_last': mask_one(s_last, A.size(0)),
        'phase': torch.tensor(1)
    })

    data.set_outputs({
        'pi': pi.clone()
    })

    return data

def floyd_warshall(A, nb_nodes, *args, **kwargs):
    """Floyd-Warshall's all-pairs shortest paths (Floyd, 1962)."""

    data = AlgorithmicData(algorithm="floyd_warshall", *args, **kwargs)

    data.set_inputs({
        'A': A.clone(),
        'adj': graph(A.clone())
    }, nb_nodes)

    D = A.clone()
    Pi = torch.zeros((A.size(0), A.size(0)), dtype=torch.int32)
    msk = graph(A.clone())

    for i in range(A.size(0)):
        for j in range(A.size(0)):
            Pi[i, j] = i

    for k in range(A.size(0)):
        prev_D = D.clone()
        prev_msk = msk.clone()

        data.increase_hints({
            'Pi_h': Pi.clone(),
            'D': prev_D.clone(),
            'msk': prev_msk.clone(),
            'k': mask_one(k, A.size(0))
        })

        for i in range(A.size(0)):
            for j in range(A.size(0)):
                if prev_msk[i, k].item() > 0 and prev_msk[k, j].item() > 0:
                    if msk[i, j].item() == 0 or prev_D[i, k].item() + prev_D[k, j].item() < D[i, j].item():
                        D[i, j] = prev_D[i, k].item() + prev_D[k, j].item()
                        Pi[i, j] = Pi[k, j].item()
                    else:
                        D[i, j] = prev_D[i, j].item()
                    msk[i, j] = 1

    data.set_outputs({
        'Pi': Pi.clone()
    })

    return data