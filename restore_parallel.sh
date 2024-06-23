path_der="derivatives/"

function restore {
    arg1="$1"
    sub_id=$(basename "$arg1" | grep -oP 'sub-\d+')
    arg2="${arg1%association_matrix*}zero_connection_nodes_${sub_id}.csv"

    echo -e "############# $sub_id, $arg1, $arg2"
    python ../code/restore_ass_mtrx.py $arg1 $arg2

}

export -f restore

find "$path_der" -type f -name '*association_matrix_*.csv' > "$path_der/ass_mtrx_files.txt"

N=100
(
for ii in $(cat "$path_der/ass_mtrx_files.txt"); do 
   ((i=i%N)); ((i++==0)) && wait
   restore "$ii" &
done
)
rm "$path_der/ass_mtrx_files.txt"