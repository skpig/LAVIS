# cport=21999
# wport=22000
# hport=8111
# model=llama2_chat_13b
# gpu_num=1
# model_path=/data2/pretrain/llama2/Llama-2-13b-chat-hf
# nohup python3 -m fastchat.serve.controller --port $cport >./logs/${model}_controller.out &
# CUDA_VISIBLE_DEVICES=$gpu_num nohup python3 -m fastchat.serve.model_worker --model-name $model --model-path $model_path --port $wport --worker-address http://localhost:$wport --controller-address http://localhost:$cport >./logs/${model}_worker.out &
# nohup python3 -m fastchat.serve.openai_api_server --host localhost --port $hport --controller-address http://localhost:$cport >./logs/${model}_sever.out &


python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-path /data2/pretrain/llama2/Llama-2-13b-chat-hf --model-name llama2_chat_13b
ython3 -m fastchat.serve.openai_api_server --host localhost --port 8000


# ps -efu|grep python3|grep -v grep|cut -c 9-16|xargs kill -9