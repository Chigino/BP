for i in {1..75}
do	
	cd $i
	echo $i
  	touch images/values
	for f in *.mp4
	do
    		base_name=""
		if [[ $f =~ (.*)[.]mp4 ]]; then
      		base_name=${BASH_REMATCH[1]}
      		#ffmpeg -i $f -r 1 -f image2 images/${BASH_REMATCH[1]}-%d.jpeg
    		fi
    		while read p
		do
			file_name=""
      			rest=""
			if [[ $p =~ (.*[.]mp4)(,.*,.*,.*,.*,*[0-9]) ]]; then
    				file_name=${BASH_REMATCH[1]}
        			rest=${BASH_REMATCH[2]}
			fi
			if [ "$f" == "$file_name" ]; then
				for j in {1..17}
        			do
          				cd images/
          				echo $base_name-$j".jpeg"$rest >> values
          				cd ..
        			done
      			fi	
		done <values	
	done
	cd ..
done
