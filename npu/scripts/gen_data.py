import numpy as np
import os


def rwkv_time_mix(B, T, C, H, data_type, input_dir, output_dir):
    N = C // H
    param_shape = (B, H, T, N)
    u_shape = (H, N)
    k = np.random.uniform(-1,1,param_shape).astype(data_type)
    v = np.random.uniform(-1,1,param_shape).astype(data_type)
    w = np.random.uniform(-1,1,param_shape).astype(data_type)
    r = np.random.uniform(-1,1,param_shape).astype(data_type)
    u = np.random.uniform(-1,1,u_shape).astype(data_type)
    o = np.zeros(param_shape).astype(data_type)
    
    # save k, v, w, r, u, o original values 
    
    k.tofile(os.path.join(input_dir,"input_k.bin"))
    v.tofile(os.path.join(input_dir,"input_v.bin"))
    w.tofile(os.path.join(input_dir,"input_w.bin"))
    r.tofile(os.path.join(input_dir,"input_r.bin"))
    u.tofile(os.path.join(input_dir,"input_u.bin"))
    o.tofile(os.path.join(input_dir,"input_o.bin"))

    np.save(os.path.join(input_dir,"input_k.bin.npy"), k)
    np.save(os.path.join(input_dir,"input_v.bin.npy"), v)
    np.save(os.path.join(input_dir,"input_w.bin.npy"), w)
    np.save(os.path.join(input_dir,"input_r.bin.npy"), r)
    np.save(os.path.join(input_dir,"input_u.bin.npy"), u)
    np.save(os.path.join(input_dir,"input_o.bin.npy"), o)

    for b in range(B):
        for h in range(H):
            print("Generating data: h=",h)
            for i in range(N):
                state = np.zeros((N), dtype=data_type)
                for t in range(T):
                    for j in range(N):
                        x = k[b, h, t, j] * v[b, h, t, i]
                        s = state[j]
                        o[b, h, t, i]+=r[b, h, t, j] * (u[h, j] * x + s)
                        state[j] = s * w[b, h, t, j] + x
    
    # output o_golden bin
    o.tofile(os.path.join(output_dir, "output_o_golden.bin"))
    np.save(os.path.join(output_dir, "output_o_golden.bin.npy"), o)
    return


if __name__=="__main__":
    B = 1
    T = 256
    C = 4096
    H = 64
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.system("rm -rf ../input_npu")
    os.system("rm -rf ../output_npu")
    os.system("mkdir ../input_npu")
    os.system("mkdir ../output_npu")
    input_dir = "../input_npu"
    output_dir = "../output_npu"
    data_type = np.float16
    rwkv_time_mix(B, T, C, H, data_type, input_dir, output_dir)
