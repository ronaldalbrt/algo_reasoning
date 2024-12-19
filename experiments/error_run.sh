python run.py --algorithms floyd_warshall --model_name floyd_warshall --version_name mpnn --processor_model mpnn --batch_size 16
python run.py --algorithms floyd_warshall --model_name floyd_warshall --version_name spectralmpnn --processor_model spectralmpnn --batch_size 16
python run.py --algorithms floyd_warshall --model_name floyd_warshall --version_name specformer --processor_model specformer --batch_size 16

python run.py --algorithms matrix_chain_order --model_name matrix_chain_order --version_name mpnn --processor_model mpnn
python run.py --algorithms matrix_chain_order --model_name matrix_chain_order --version_name spectralmpnn --processor_model spectralmpnn
python run.py --algorithms matrix_chain_order --model_name matrix_chain_order --version_name specformer --processor_model specformer

python run.py --algorithms graham_scan --model_name graham_scan --version_name spectralmpnn --processor_model spectralmpnn

python run.py --algorithms jarvis_march --model_name jarvis_march --version_name spectralmpnn --processor_model spectralmpnn
python run.py --algorithms jarvis_march --model_name jarvis_march --version_name specformer --processor_model specformer