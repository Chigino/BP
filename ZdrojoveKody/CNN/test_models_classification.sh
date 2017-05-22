for f in *model
do
	echo $f >> res
	python test_net_classification.py $f | tail -5 >> res
done
