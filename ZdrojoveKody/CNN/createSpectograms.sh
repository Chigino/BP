for f in *.wav
do
	# extracting base name from file name
	train_name=""
	if [[ $f =~ (.*[.][0-9]*).wav ]]; then
		train_name=${BASH_REMATCH[1]}
	fi

	echo $f
	python spectograms.py $train_name
done
