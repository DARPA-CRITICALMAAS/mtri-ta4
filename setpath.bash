SCRIPTDIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd )"

DEFAULT_PATH=~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
export QGIS_PLUGINPATH="$DEFAULT_PATH;${SCRIPTDIR}/plugins/"
#export PYDEVD_USE_CYTHON=NO
#export PYDEVD_USE_FRAME_EVAL=NO
