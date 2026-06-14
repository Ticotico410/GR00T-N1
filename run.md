## H200 服务器训练

### 启动训练脚本：
```bash
chmod +x run.sh

nohup bash run.sh > train.log 2>&1 < /dev/null &

echo $! > train.pid

disown
```

---

### 查看运行日志
```bash
cat train.pid
ps -fp $(cat train.pid)
tail -f train.log
```

---

### 杀死进程
```bash
pkill -9 -f "/root/shanghai/ycb/GR00T-N1/scripts/gr00t_finetune.py"
```

chmod +x run_1.sh

nohup bash run_1.sh > train_1.log 2>&1 < /dev/null &

echo $! > train_1.pid

disown

tail -f train_1.log