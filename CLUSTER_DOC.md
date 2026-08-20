登录与存储
ssh tomyyc@ilc.stanford.edu        # 校外先连 Stanford VPN

路径
特性
用途
/sailhome/$USER
20GB,NFS
只放配置,别放代码/数据/venv
/dfs/user/$USER
2TB,慢 NFS,全节点共享
代码仓库、数据集、checkpoint
/lfs/local/0/$USER
大容量快 SSD,每节点独立
venv、pip/HF 缓存(换节点要重建)

硬规矩(违反直接被拒/被杀)
每个任务必须带 --account=infolab;
login 节点不跑任何计算;
交互任务必须 --qos=il-interactive(12h 上限,优先级最高);批处理默认 --qos=il(7 天);长任务 --qos=il-lo(21 天);
CPU-only 任务在 ampere 节点最多 8 CPU/32G,大 CPU 任务用 --partition=il-cpu;
每用户 GPU 配额:A100=10 块,B200=2 块(DenyOnLimit:单任务超配额直接拒收)。


资源侦察
# 各节点卡型/占用(最常用)
sinfo -p il -O NodeHost:.20,Gres:.40,GresUsed:.60

# 节点是否维护状态(idle/mix 可用,drain/down 别碰)
sinfo -p il -O NodeHost:.14,StateLong:.16,Reason:.40

# 查自己的 QoS 配额上限
sacctmgr show qos il format=Name%12,Flags%24,MaxTRESPerUser%70

# 查自己的 account/QoS 归属
sacctmgr show user $USER withassoc format=user,account%20,qos%30

提交任务
# 交互式 debug(1 块 A100,4 小时)
srun --account=infolab --partition=il --qos=il-interactive \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=100G --time=04:00:00 --pty bash

# 我们仓库的标准流程(在 /dfs/user/$USER/LLM_training 下)
PREP=$(sbatch --parsable scripts/ilc_prepare.sbatch)                  # 数据准备(CPU)
sbatch --dependency=afterok:$PREP scripts/ilc_train.sbatch configs/attnres_124m.yaml   # 依赖链训练
sbatch scripts/ilc_train.sbatch configs/baseline_124m.yaml train.seed=42 train.run_name=baseline_124m_s42  # 带覆盖参数

# 命令行覆盖脚本里的资源(比如换卡型/数量;世界大小脚本会自动跟随)
sbatch --gres=gpu:b200:2 scripts/ilc_train.sbatch configs/attnres_124m.yaml train.grad_accum_steps=8

监控与管理
squeue -u $USER                                                    # 我的队列
squeue -j <jobid> --start                                          # 预估开始时间
squeue -u $USER -O jobid:10,state:10,starttime:22,endtime:22       # 开始/结束时间总览
tail -f slurm-<jobid>.out                                          # 任务 stdout/stderr
tail -3 out/<run_name>/log.jsonl                                   # 训练真心跳(每步落盘)
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList    # 已结束任务的死因
scancel <jobid> [<jobid>...]                                       # 取消
scontrol update JobId=<jobid> Dependency=afterok:<new_prep_id>     # 改依赖指向


srun --account=infolab --partition=il --qos=il-interactive \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=100G --time=04:00:00 --pty bash


# 不要 GPU,纯 CPU 调试(注意 ampere 节点上限 8 CPU/32G)
srun --account=infolab --partition=il --qos=il-interactive \
  --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash

# 多卡交互(配额内,A100 最多 10 块)
srun ... --gres=gpu:a100:2 ...

# 冷知识:要 1 块 B200 基本秒到(没人用得了那台机器)——单卡跑没问题,
# 只是别在上面跑多卡 NCCL(会挂),且要用 cu13 的 venv
srun ... --gres=gpu:b200:1 ...




