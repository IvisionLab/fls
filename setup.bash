SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
if ! test -n "$PYTHONPATH"; then
  PYTHONPATH=$SCRIPT_DIR
else
  if ! [[ :$PYTHONPATH: == *:"$SCRIPT_DIR":* ]] ; then
    PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH
  fi
fi
export PYTHONPATH