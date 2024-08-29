export CUDA_VISIBLE_DEVICES=0
# Accuracy : 0.9882, Precision : 0.9679, Recall : 0.9857, F-score : 0.9767
python main.py --anormly_ratio 1.5 --num_epochs 3    --batch_size 256  --mode train --dataset PSM  --data_path PSM --input_c 25    --output_c 25  --loss_fuc MSE  --win_size 60 
#python main.py --anormly_ratio 1.5  --num_epochs 10       --batch_size 256     --mode test    --dataset PSM   --data_path PSM  --input_c 25    --output_c 25  --loss_fuc MSE  --win_size 60 

#Accuracy : 0.9875, Precision : 0.8087, Recall : 0.9194, F-score : 0.8605
#python main.py --anormly_ratio 1.1 --num_epochs 3   --batch_size 256  --mode train --dataset SMD  --data_path SMD   --input_c 38   --output_c 38  --loss_fuc MSE  --win_size 105
#python main.py --anormly_ratio 1.1 --num_epochs 10   --batch_size 256  --mode test    --dataset SMD   --data_path SMD     --input_c 38      --output_c 38   --loss_fuc MSE   --win_size 105

# Accuracy : 0.9916, Precision : 0.9344, Recall : 1.0000, F-score : 0.9661
#python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode train --dataset SWAT  --data_path SWAT  --input_c 51    --output_c 51  --loss_fuc MSE --win_size 105
#python main.py --anormly_ratio 1  --num_epochs 10   --batch_size 128     --mode test    --dataset SWAT   --data_path SWAT  --input_c 51    --output_c 51   --loss_fuc MSE --win_size 105

# Accuracy : 0.9888, Precision : 0.9159, Recall : 0.9882, F-score : 0.9507
#python main.py --anormly_ratio 1.1 --num_epochs 3   --batch_size 64  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 90 
#python main.py --anormly_ratio 1.1  --num_epochs 10     --batch_size 64    --mode test    --dataset MSL   --data_path MSL --input_c 55    --output_c 55   --win_size 90 

#python main.py --anormly_ratio $anormly_ratio --num_epochs 3   --batch_size 256  --mode train --dataset SMAP  --data_path SMAP --input_c 25    --output_c 25  --loss_fuc MSE --win_size 105 --e_layers $layer
#python main.py --anormly_ratio $anormly_ratio  --num_epochs 10   --batch_size 256     --mode test    --dataset SMAP   --data_path SMAP  --input_c 25    --output_c 25   --loss_fuc MSE --win_size 105 --e_layers $layer

