#python bagging_inference_nlp.py --ds=sst2 --m=1 --bagging=False &&
#python bagging_inference_nlp.py --ds=sst2 --m=2 --bagging=False &&
#python bagging_inference_nlp.py --ds=sst2 --m=5 --bagging=False &&
#python bagging_inference_nlp.py --ds=sst2 --m=10 --bagging=False &&
#python bagging_inference_nlp.py --ds=sst2 --m=20 --bagging=False &&
#python bagging_inference_nlp.py --ds=sst2 --m=30 --bagging=False &&
#python bagging_inference_nlp.py --ds=sst2 --m=50 --bagging=False &&


python bagging_inference_nlp.py --ds=sst2 --m=1 --bagging=True &&
python bagging_inference_nlp.py --ds=sst2 --m=2 --bagging=True &&
python bagging_inference_nlp.py --ds=sst2 --m=5  --bagging=True &&
python bagging_inference_nlp.py --ds=sst2 --m=10 --bagging=True &&
python bagging_inference_nlp.py --ds=sst2 --m=20 --bagging=True &&
python bagging_inference_nlp.py --ds=sst2 --m=30  --bagging=True &&
python bagging_inference_nlp.py --ds=sst2 --m=50 --bagging=True
