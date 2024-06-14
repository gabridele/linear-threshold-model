path_der="derivatives/"

function simulation {
 input="$1"

 python ../code/linear_threshold_model_association.py "$input"
}

export -f simulation

find "$path_der" -type f -name '*5000000mio_connectome.csv' > "$path_der/connectome_files.txt"

N=10
(
for ii in $(cat "$path_der/connectome_files.txt"); do 
   ((i=i%N)); ((i++==0)) && wait
   simulation "$ii" &
done
)
rm "$path_der/connectome_files.txt"