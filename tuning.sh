for lr in 0.0001 0.0002 0.0005 0.002 0.001 0.005
do
 for beta1 in 0.5 0.6 0.7 0.8 0.9
 do
  for beta2 in 0.999 0.99 0.9 0.995
  do
   python main.py --src ../ear-binary --lr $lr --beta_1 $beta1 --beta_2 $beta2 --nb_epochs 10 --start_fold 0 --end_fold 1
  done
 done
done
