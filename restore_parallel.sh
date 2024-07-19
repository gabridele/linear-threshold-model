path_der="derivatives/"

function restore {
    arg1="$1"
    sub_id=$(basename "$arg1" | grep -oP 'sub-\d+')
    n_seed=$(basename "$arg1" | awk -F '_' '{print $NF}' | cut -d 's' -f 1)
    arg2="${arg1%association_matrix*}zero_connection_nodes_${sub_id}_${n_seed}seeds.csv"
    
    echo -e "############# $sub_id, $arg1, $arg2"
    python ../code/linear-threshold-model/restore_zscore_assoc_mtrx.py $arg1 $arg2 $n_seed

}

export -f restore

find "$path_der" -type f -name '*association_matrix_*_30seeds.csv' > "$path_der/ass_mtrx_files.txt"

N=140
(
for ii in $(cat "$path_der/ass_mtrx_files.txt"); do 
   ((i=i%N)); ((i++==0)) && wait
   restore "$ii" &
done
)
rm "$path_der/ass_mtrx_files.txt"