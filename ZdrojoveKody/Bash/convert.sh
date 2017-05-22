cd sound_train

#for f in -- *.mp3
#do
#	name=""
#	skip=false
#	if [[ $f =~ (.*)[.] ]]; then
#		name=${BASH_REMATCH[1]}
#	fi
#
#	mpg123 -w $name.wav -- $f
#	echo train $f
#done

cd ../sound_test
skip=true
for f in *.mp3
do
	
	

	name=""
	if [[ $f =~ (.*)[.] ]]; then
		name=${BASH_REMATCH[1]}
	fi

	mpg123 -w $name.wav -- $f
	echo test $f
done