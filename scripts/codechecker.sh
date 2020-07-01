#! /bin/bash
echo "==================================================================="
echo "You can chosse argmuments \"all\" for full code error check or \"format\" for google style check"
echo "In addtion, more flags for error check to replace \"all\" : 
                    * warning
                                  Enable warning messages
                     * style
                                  Enable all coding style checks. All messages
                                  with the severities 'style', 'performance' and
                                  'portability' are enabled.
                     * performance
                                  Enable performance messages
                     * portability
                                  Enable portability messages
                     * information
                                  Enable information messages
                     * unusedFunction
                                  Check for unused functions. It is recommend
                                  to only enable this when the whole program is
                                  scanned.
                    * missingInclude
                                  Warn if there are missing includes. For
                                  detailed information, use '--check-config'.
                    Several ids can be given if you separate them with
                    commas.
"
echo "==================================================================="
echo "checking your modifications in current repo..."
sleep 2
FILE_LIST=$(git status -s) 

NEXT_IS_FILE=false
FILE_STATE=""
LIST_FOR_CHECK=""


for file in $FILE_LIST
do
    if $NEXT_IS_FILE; then
        if [[ "$FILE_STATE" != "D" ]];then 
            if [[ $file == *.cpp ]] || [[ $file == *.h ]] || [[ $file == *.hpp ]];then
                #echo "needed for check $FILE_STATE: $file"
                LIST_FOR_CHECK="$LIST_FOR_CHECK $file"
            fi
        fi
        NEXT_IS_FILE=false
    else 
        FILE_STATE=$file
        NEXT_IS_FILE=true
    fi
done

if [ "$1" = "format" ];then
    echo "--------- checking with google code style -----------"
    if ! cpplint --version; then
        echo "cpplint NOT found, install it !!!!"
        exit 0
    fi
    cpplint $LIST_FOR_CHECK
else
    if [ "$1" = "" ];then
         CHECK_FLAG="all"
    else 
        CHECK_FLAG=$1
    fi
    echo "--------- using static code checker -----------"
    if ! cppcheck --version; then
        echo "cppcheck NOT found, install it !!!!"
        exit 0
    fi
    INCLUDE_DIRS=$(find modules -name "include" -type d)
    INCLUDE_FLAG=""
    for dir in $INCLUDE_DIRS
    do
        INCLUDE_FLAG="$INCLUDE_FLAG -I $dir"
    done
    #echo $INCLUDE_FLAG
    cppcheck --language=c++ --std=c++11 --enable=$CHECK_FLAG $INCLUDE_FLAG $LIST_FOR_CHECK
fi
