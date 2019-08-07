CORES_PER_SOCK=`lscpu |grep "^Core(s) per socket:" |awk -F: '{print $2}' |awk '{$1=$1};1'`

echo "Set OMP_NUM_THREADS="$CORES_PER_SOCK
export OMP_NUM_THREADS=$CORES_PER_SOCK
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
