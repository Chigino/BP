cd train/
old=""

for f in *.jpeg
do
	# extracting base name from file name
	train_name=""
	if [[ $f =~ (.*[.][0-9]*)- ]]; then
		train_name=${BASH_REMATCH[1]}
	fi
	# if the base name is same skip cycle
	if [ "$train_name" == "$old" ] 
	then
		continue;
	fi	
	# getting values for image
	while read p
		do
		file_name=""
		value=""
		if [[ $p =~ (.*)[.]mp4,(.*,.*,.*,.*,*[0-9]) ]]; then
 			file_name=${BASH_REMATCH[1]}
			value=${BASH_REMATCH[2]}
		fi
		if [ "$file_name" == "$train_name" ]
		then
			break
		fi
	done <values

	echo $train_name

	python getImageActivations.py $train_name $value > tmp

	# marking name of last process image group
	old=$train_name
done
