echo "Evaluation for suno and udio datasets with musicnn and svc"
/home/laura/aimir/musicnn_env/bin/python /home/laura/aimir/src/evaluate_ai_detector.py --ai-folders suno udio --embedding-type musicnn --classifier-type svc
echo "Evaluation for suno and udio datasets with musicnn and rf"
/home/laura/aimir/musicnn_env/bin/python /home/laura/aimir/src/evaluate_ai_detector.py --ai-folders suno udio --embedding-type musicnn --classifier-type rf
echo "Evaluation for suno and udio datasets with musicnn and dnn"
/home/laura/aimir/musicnn_env/bin/python /home/laura/aimir/src/evaluate_ai_detector.py --ai-folders suno udio --embedding-type musicnn --classifier-type dnn
echo "Evaluation for suno and udio datasets with clap-laion-music and svc"
/home/laura/aimir/embeddings_env/bin/python /home/laura/aimir/src/evaluate_ai_detector.py --ai-folders suno udio --embedding-type clap-laion-music --classifier-type svc
echo "Evaluation for suno and udio datasets with clap-laion-music and rf"
/home/laura/aimir/embeddings_env/bin/python /home/laura/aimir/src/evaluate_ai_detector.py --ai-folders suno udio --embedding-type clap-laion-music --classifier-type rf
echo "Evaluation for suno and udio datasets with clap-laion-music and dnn"
/home/laura/aimir/embeddings_env/bin/python /home/laura/aimir/src/evaluate_ai_detector.py --ai-folders suno udio --embedding-type clap-laion-music --classifier-type dnn
