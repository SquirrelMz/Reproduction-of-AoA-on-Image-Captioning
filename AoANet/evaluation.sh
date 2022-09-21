# AoANet after reinforcement learning
#CUDA_VISIBLE_DEVICES=7 python eval.py --model log/log_aoanet_rl/model.pth --infos_path log/log_aoanet_rl/infos_aoanet.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test > vis/scores/score_aoa_rl.txt

# AoANet without reinforcement learning
CUDA_VISIBLE_DEVICES=7 python eval.py --model log/log_aoanet/model.pth --infos_path log/log_aoanet/infos_aoanet.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test > vis/scores/score_aoa_worl.txt

# baseline after reinforcement learning
#CUDA_VISIBLE_DEVICES=7 python eval.py --model log/log_base_rl/model.pth --infos_path log/log_base_rl/infos_base.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test > vis/scores/score_base_rl.txt

# baseline without reinforcement learning
#CUDA_VISIBLE_DEVICES=7 python eval.py --model log/log_base/model.pth --infos_path log/log_base/infos_base.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test > vis/scores/score_base_worl.txt
