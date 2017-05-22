cd video

for f in *.mp4
do
	name=""
	if [[ $f =~ (.*)[.] ]]; then
		name=${BASH_REMATCH[1]}
	fi

	ffmpeg -i $f -ac 1 wav/$name.wav >> /dev/null
	echo $f
done


