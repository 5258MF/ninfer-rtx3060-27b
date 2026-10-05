@echo off
echo =======================================================
echo Downloading Qwen3.6-35B-A3B Model (groupwise-int)...
echo File Size: ~19GB
echo Please wait, this may take a while depending on your network.
echo =======================================================
curl -L -C - -o qwen3_6_35b_a3b.ninfer https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/main/qwen3_6_35b_a3b.ninfer
echo =======================================================
echo Download complete!
echo NOTE: the verified 260k profile uses --kv-dtype rk4v4-e8
echo (runtime KV quantization; works with this public artifact).
echo =======================================================
pause
