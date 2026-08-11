"""
seior
向量操作，计算
状态矩阵：n*1维，0，1，2，3.....表示状态
"""
import cupy as cp
import time
import pickle
import json
from pathlib import Path
import networkx as nx
import os
import numpy as np
import gc
import scipy.sparse as sp
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_edge_chunk(file_path, skip_lines, chunk_size):
    """
    从 .edgelist 文件分块读取边数据
    """
    edges = []
    with open(file_path, 'r') as f:
        # 跳过已读行
        for _ in range(skip_lines):
            next(f)
        for _ in range(chunk_size):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
    return np.array(edges, dtype=np.int32)

def get_num_nodes_from_file(file_path):
    max_node = -1
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u > max_node:
                max_node = u
            if v > max_node:
                max_node = v
    return max_node + 1
def build_gpu_sparse_adj_from_file(file_path, directed = False,chunk_size=5000000, n_know = True):
    print("扫描文件获取节点数...")
    if n_know:
        n = 10**8
    else:
        n = get_num_nodes_from_file(file_path)
    print(f"节点总数: {n}")

    # 初始化空的CSR矩阵（用于累加所有块）
    adjacency_matrix = cp.sparse.csr_matrix((n, n), dtype=cp.float32)
    skip_lines = 0
    total_edges_processed = 0

    while True:
        # 1. 读取单块边数据
        edges_np = read_edge_chunk(file_path, skip_lines, chunk_size)
        if len(edges_np) == 0:
            break
        skip_lines += len(edges_np)
        total_edges_processed += len(edges_np)
        if directed:
            expanded_edges = edges_np
        else:
            # 2. 扩展双向边（单块内处理）
            expanded_edges = np.vstack((edges_np, edges_np[:, [1, 0]]))

        rows_np = expanded_edges[:, 0]
        cols_np = expanded_edges[:, 1]
        data_np = np.ones(len(rows_np), dtype=np.float32)

        # 3. 单块转为COO→再转为CSR（避免大临时数组）
        # 先用NumPy创建COO（减少GPU显存占用）
        coo_np = sp.coo_matrix(
            (data_np, (rows_np, cols_np)),
            shape=(n, n)
        )
        # 转GPU的CSR（单块数据量小，转换时临时内存可控）
        chunk_csr = cp.sparse.csr_matrix(coo_np)

        # 4. 累加当前块到总矩阵（CSR合并）
        adjacency_matrix = adjacency_matrix + chunk_csr

        print(f"已处理 {total_edges_processed} 条边")

        # 5. 强制释放当前块的显存和内存
        del edges_np, expanded_edges, rows_np, cols_np, data_np, coo_np, chunk_csr
        gc.collect()
        cp._default_memory_pool.free_all_blocks()  # 释放CuPy缓存

    print("邻接矩阵构造完成！")
    return adjacency_matrix, n
class GenericStatePropagationGPU:
    def __init__(self, graph, num_states, n, A):
        # ---------- 构建稀疏邻接 ----------
        '''
        edges = list(graph.edges())
        if not graph.is_directed():
            edges = edges + [(v, u) for u, v in edges]
        rows = cp.array([u for u, v in edges], dtype=cp.int32)
        cols = cp.array([v for u, v in edges], dtype=cp.int32)
        data = cp.ones(len(edges), dtype=cp.int8)
        self.n = graph.number_of_nodes()
        self.A = sp.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
        '''

        self.A = A
        self.n = n
        self.num_states = num_states
        # self.state = cp.zeros(self.n, dtype=cp.int8)
        # Rw = cp.array([0.98, 0.0, 0.02, 0.0, 0.0], dtype=cp.float32)
        # self.state = cp.random.choice(self.num_states, size=self.n, p=Rw).astype(cp.int8)
        rand = cp.random.rand(self.n)
        self.state = cp.zeros(self.n, dtype=cp.int8)
        self.state[rand >= 0.98] = 2


        self.rules = []
        '''参数设置'''
        self.alpha1 = 0.2
        self.alpha2 = 0.5
        self.alpha3 = 0.2
        self.eta0 = 0.2
        self.eta1 = 0.4
        self.eta2 = 0.4
        self.beta1 = 0.6
        self.beta2 = 0.6
        self.beta3 = 0.3
        self.delta = 0.1

        self.I_num = cp.full((self.n,), 0, dtype=cp.int8)
        self.num_O = cp.full((self.n,), 0, dtype=cp.int8)
        self.p_er = cp.full((self.n,), self.delta, dtype=cp.float32)
        self.p_ir1 = cp.full((self.n,), self.delta, dtype=cp.float32)
        self.p_or = cp.full((self.n,), self.delta, dtype=cp.float32)
        def p_SE(model):
            # 无知者--犹豫者
            return 1.0 - cp.power(1.0 - model.alpha1, model.I_num)

        def p_SI(model):
            # 无知者--传播者
            return 1.0 - cp.power(1.0 - model.alpha2, model.I_num)

        def p_SO(model):
            # 无知者--辟谣者
            return 1.0 - cp.power(1.0 - model.eta0, model.I_num)

        def p_SR(model):
            # 无知者--免疫者
            return 1.0 - cp.power(1.0 - model.delta, model.I_num)

        def p_EI(model):
            # 犹豫者--传播者
            return 1.0 - cp.power(1.0 - model.alpha3, model.I_num)
        def p_EO(model):
            # 犹豫者--辟谣者
            return 1.0 - cp.power(1.0 - model.eta1, model.num_O)
        def p_ER(model):
            # 犹豫者--免疫者
            return self.p_er

        def p_IO(model):
            # 传播者--辟谣者
            return 1.0 - cp.power(1.0 - model.eta2, model.num_O)

        def p_IR1(model):
            # 传播者--免疫者
            return self.p_ir1
        def p_IR2(model):
            # 传播者--免疫者
            num_IO = model.I_num + model.num_O
            return 1.0 - cp.power(1.0 - model.beta1, num_IO)
        def p_IR3(model):
            # 传播者--免疫者
            return 1.0 - cp.power(1.0 - model.beta2, model.num_O)
        def p_IR4(model):
            # 传播者--免疫者
            return 1.0 - cp.power(1.0 - model.beta3, model.num_O)
        def p_OR(model):
            # 辟谣者--免疫者
            return self.p_or

        self.add_rule(0, 1, p_SE)
        self.add_rule(0, 2, p_SI)
        self.add_rule(0, 3, p_SO)
        self.add_rule(0, 4, p_SR)
        self.add_rule(1, 2, p_EI)
        self.add_rule(1, 3, p_EO)
        self.add_rule(1, 4, p_ER)
        self.add_rule(2, 3, p_IO)
        self.add_rule(2, 4, p_IR1)
        self.add_rule(2, 4, p_IR2)
        self.add_rule(2, 4, p_IR3)
        self.add_rule(2, 4, p_IR4)
        self.add_rule(3, 4, p_OR)

        self.s_num = []
        self.e_num = []
        self.i_num = []
        self.o_num = []
        self.r_num = []
        each_state_sums = cp.bincount(self.state, minlength=self.num_states)
        self.s_num.append(each_state_sums[0])
        self.e_num.append(each_state_sums[1])
        self.i_num.append(each_state_sums[2])
        self.o_num.append(each_state_sums[3])
        self.r_num.append(each_state_sums[4])

        self.Rules_num = len(self.rules)

    # ---------- 添加规则 ----------
    def add_rule(self, from_state, to_state, prob_func):

        self.rules.append({
            "from": from_state,
            "to": to_state,
            "prob": prob_func
        })

    # ---------- 一步传播 ----------
    def step(self):
        I_mask = (self.state == 2).astype(cp.int8)
        self.I_num = self.A @ I_mask
        O_mask = (self.state == 3).astype(cp.int8)
        self.num_O = self.A @ O_mask

        hit = cp.zeros((self.Rules_num, self.n), dtype=cp.bool_)
        P = cp.zeros((self.Rules_num, self.n), dtype=cp.float32)
        T = cp.empty(self.Rules_num, dtype=cp.int8)
        for r, rule in enumerate(self.rules):
            f = rule["from"]
            t = rule["to"]
            mask = (self.state == f)
            p = rule["prob"](self)  # (N,)
            rand = cp.random.rand(self.n)
            hit_r = mask & (rand < p)
            hit[r, hit_r] = True
            P[r, hit_r] = p[hit_r]
            T[r] = t

        best_rule = cp.argmax(P, axis=0)
        best_p = P[best_rule, cp.arange(self.n)]
        # best_p = cp.max(P, axis=0)  # (N,)
        do_trans = best_p > 0
        new_state = self.state.copy()
        new_state[do_trans] = T[best_rule[do_trans]]
        self.state = new_state

        each_state_sums = cp.bincount(self.state, minlength=self.num_states)
        self.s_num.append(each_state_sums[0])
        self.e_num.append(each_state_sums[1])
        self.i_num.append(each_state_sums[2])
        self.o_num.append(each_state_sums[3])
        self.r_num.append(each_state_sums[4])

    def states_num(self):
        return self.s_num, self.e_num, self.i_num, self.o_num, self.r_num

def spread_main(path, large = False):
    if large:
        adjacency_matrix, nodes_num = build_gpu_sparse_adj_from_file(path, directed=False, n_know=True)
        graph = 0
    else:
        with open(path, "rb") as f:
            graph = pickle.load(f)
        edges = list(graph.edges())
        if graph.is_directed():
            expanded_edges = edges
        else:
            expanded_edges = edges + [(v, u) for u, v in edges]
        rows = cp.array([edge[0] for edge in expanded_edges], dtype=cp.int32)
        cols = cp.array([edge[1] for edge in expanded_edges], dtype=cp.int32)
        data = cp.ones(len(expanded_edges), dtype=cp.float32)
        nodes_num = graph.number_of_nodes()
        adjacency_matrix = cp.sparse.csr_matrix((data, (rows, cols)), shape=(nodes_num, nodes_num))
    all_s = []
    all_e = []
    all_i = []
    all_o = []
    all_r = []
    all_time = []
    for j in range(20):
        sir = GenericStatePropagationGPU(graph, 5, nodes_num, adjacency_matrix)
        time1 = time.time()
        for i in range(100):
            sir.step()
        s_num, e_num, i_num, o_num, r_num = sir.states_num()
        time2 = time.time()

        all_s.append([v.item() for v in s_num])
        all_e.append([v.item() for v in e_num])
        all_i.append([v.item() for v in i_num])
        all_o.append([v.item() for v in o_num])
        all_r.append([v.item() for v in r_num])

        all_time.append(time2 - time1)
    data_result = {
        'n': nodes_num,
        's_num': all_s,
        'e_num': all_e,
        'i_num': all_i,
        'o_num': all_o,
        'r_num': all_r,
        'time': all_time
    }
    return data_result

def spread_real_main(path, large = False, directed = False):
    if large:
        if directed:
            adjacency_matrix, nodes_num = build_gpu_sparse_adj_from_file(path, directed=True, n_know=False)
        else:
            adjacency_matrix, nodes_num = build_gpu_sparse_adj_from_file(path, directed=False, n_know=False)
        graph = 0
    else:
        if directed:
            graph = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph(), data=False)
        else:
            graph = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph(), data=False)
        graph = nx.convert_node_labels_to_integers(graph, first_label=0)
        edges = list(graph.edges())
        if graph.is_directed():
            expanded_edges = edges
        else:
            expanded_edges = edges + [(v, u) for u, v in edges]
        rows = cp.array([edge[0] for edge in expanded_edges], dtype=cp.int32)
        cols = cp.array([edge[1] for edge in expanded_edges], dtype=cp.int32)
        data = cp.ones(len(expanded_edges), dtype=cp.float32)
        nodes_num = graph.number_of_nodes()
        adjacency_matrix = cp.sparse.csr_matrix((data, (rows, cols)), shape=(nodes_num, nodes_num))
    all_s = []
    all_e = []
    all_i = []
    all_o = []
    all_r = []
    all_time = []
    for j in range(20):
        sir = GenericStatePropagationGPU(graph, 5, nodes_num, adjacency_matrix)
        time1 = time.time()
        for i in range(100):
            sir.step()
        s_num, e_num, i_num, o_num, r_num = sir.states_num()
        time2 = time.time()
        all_s.append([v.item() for v in s_num])
        all_e.append([v.item() for v in e_num])
        all_i.append([v.item() for v in i_num])
        all_o.append([v.item() for v in o_num])
        all_r.append([v.item() for v in r_num])
        all_time.append(time2 - time1)
    data_result = {
        'n': nodes_num,
        's_num': all_s,
        'e_num': all_e,
        'i_num': all_i,
        'o_num': all_o,
        'r_num': all_r,
        'time': all_time
    }
    return data_result
def main_2_7():
    for n in [2, 3, 4, 5, 6, 7]:
        for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
            path = f'./network/small_world_network/small_world_network_10**{n}_{p}.gpickle'
            data = spread_main(path)
            with open(f"./result/small_world_network/result_framework_new_state_seior_20_step_{10**n}_{p}.json", "w",
                      encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f'framework {10**n}个节点p为{p}的网络结果已保存')

        for seed in [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]:
            path = f'./network/scale_free_network/scale_free_net_10**{n}_seed{seed}.gpickle'
            data = spread_main(path)
            with open(f"./result/scale_free_network/result_framework_new_state_seior_20_step_{10**n}_seed{seed}.json", "w",
                      encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f'framework {10**n}个节点seed为{seed}的网络结果已保存')
def main_8():
    for p in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        path = f"./network/small_world_network_10**8_{p}.edgelist"
        data = spread_main(path, True)
        with open(f"./result/small_world_network/result_framework_new_state_seior_20_step_{10 ** 8}_{p}.json", "w",
                  encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'framework {10 ** 8}个节点p为{p}的网络结果已保存')
    for seed in [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]:
        path = f'./network/scale_free_net_10**8_seed{seed}.edgelist'
        data = spread_main(path, True)
        with open(f"./result/scale_free_network/result_framework_new_state_seior_20_step_{10 ** 8}_seed{seed}.json", "w",
                  encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'framework {10 ** 8}个节点seed为{seed}的网络结果已保存')
def get_file_info(folder_path):
    """获取文件夹中所有文件的文件名和绝对路径"""
    folder = Path(folder_path)
    file_info_list = []
    for item in folder.glob("*"):
        if item.is_file():
            file_info_list.append({
                "file_name": item.name,
                "file_path": str(item.resolve())
            })
    return file_info_list
def real_main():
    target_folder = "./network/real_network/directed"
    files = get_file_info(target_folder)
    for f in files:
        print(f"文件名: {f['file_name'][:-9]}")
        print(f"路径: {f['file_path']}\n")
        path = f['file_path']
        filename = f['file_name'][:-9]

        if filename == 'soc-LiveJournal1' or filename == 'ego-gplus':
            data = spread_real_main(path, True, True)
        else:
            data = spread_real_main(path, False, True)
        with open(f"./result/real_network/directed/result_framework_new_state_seior_20_step_{filename}.json", "w",
                  encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'framework {filename}的网络结果已保存')
    target_folder = "./network/real_network/undirected"
    files = get_file_info(target_folder)
    for f in files:
        print(f"文件名: {f['file_name'][:-9]}")
        print(f"路径: {f['file_path']}\n")
        path = f['file_path']
        filename = f['file_name'][:-9]
        if filename == 'com-orkut':
            data = spread_real_main(path, True, False)
        else:
            data = spread_real_main(path, False, False)
        with open(f"./result/real_network/undirected/result_framework_new_state_seior_20_step_{filename}.json", "w",
                  encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'framework{filename}的网络结果已保存')

if __name__ == '__main__':
    main_2_7()
    real_main()
    main_8()