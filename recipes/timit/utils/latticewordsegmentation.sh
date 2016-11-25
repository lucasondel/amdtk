#!/usr/bin/env bash
#####################################################################################################################################
### Call: StartSim_htk_lattice_fixed_noref.bash FileListPath Outputdir RescoredLatticesDir KnownN UnkN NumIter AmScale PruneFactor ##
### e.g.: ./StartSim_htk_lattice_fixed_noref.bash WSJ_AMDTK_Lattices/wsj_amdtk_flist.txt OUTDIR RESLATDIR 1 2 200 1 16             ##
###                                                                                                                      	   ##
### Note this scipt mainly demonstrates the processing from HTK slf lattices.                                             	   ##
###                                                                                                                        	   ##
### Runs segmentation on htk lattice input with word LM order KnownN, character LM  order UnkN                     	           ##
### for NumIter iterations. Segmentation is done by Gibbs sampling for 150 iterations, then by                                     ##
### Viterby decoding for another 25 iterations. From iteration 175 on the character language model     	                           ##
### is deactivated by setting d and Theta to zero for the 0-gram word language model. The language            	                   ##
### model is always estimated by Gibbs sampling. Estimation of the word length distribution and                                    ##
### correction factors is not done. The htk lattice are pruned with the given pruning factor.                                      ##
###                                                                                                                                ##
### For input file format see WSJCAM0_WSJ0+1_Cross_Lattice/WSJCAM0_Phoneme_Lattice.htk.txt (file list)                             ##
### WSJCAM0_WSJ0+1_Cross_Lattice/htk/*.lat (the htk slf lattices)                                                                  ##
### WSJCAM0_WSJ0+1_Cross_Lattice/WSJCAM0_Phoneme_Lattice.htk.txt.ref (the reference word transcription)                            ##
#####################################################################################################################################

# "$PWD/utils/StartSim_htk_lattice_fixed_noref.bash ${setup} ${file_list} ${outbase} ${NumIter} \$ITEM1" \

set -e

source ${1}

### parse some parameters ###
FileListPath="${2}"

#Item1: KnownN/UnkN/AddCharN/AmScale/PruneFactor
Known_N=$(echo ${5}|cut -d '/' -f 1)
Unk_N=$(echo ${5}|cut -d '/' -f 2)
AddChar_N=$(echo ${5}|cut -d '/' -f 3)
Am_Scale=$(echo ${5}|cut -d '/' -f 4)
Prune_Factor=$(echo ${5}|cut -d '/' -f 5)
Output_Directory_Basename=${3}/${AddChar_N}_${Am_Scale}_${Prune_Factor}

### Global Options ###
KnownN="-KnownN ${Known_N}"                                                                     # The n-gram length of the word language model (-KnownN N (1))
UnkN="-UnkN ${Unk_N}"                                                                         # The n-gram length of the character language model (-UnkN N (1))
AddCharN="-AddCharN ${AddChar_N}"
NumIter="-NumIter ${4}"	                                                                  # Maximum number of iterations (-NumIter N (0))
OutputDirectoryBasename="-OutputDirectoryBasename ${Output_Directory_Basename}/"                                  # The basename for result outpt Directory (Parameter: -OutputDirectoryBasename OutputDirectoryBasename ())
WordLengthModulation='-WordLengthModulation -1'                                           # Set word length modulation. -1: off, 0: automatic, >0 set mean word length (-WordLengthModulation WordLength (-1))
UseViterby='-UseViterby 151'
DeactivateCharacterModel='-DeactivateCharacterModel 175'
NoThreads="-NoThreads ${latticewordsegmentation_threads}"

### Options when using text ###
InputFilesList="-InputFilesList ${FileListPath}"                                          # A list of input files, one file per line. (-InputFilesList InputFileListName (NULL))

### additional Options when using lattices ###
InputType='-InputType fst'                                                                # The type of the input (-InputType [text|fst] (text))
PruneFactor="-PruneFactor ${Prune_Factor}"                                                           # Prune paths in the input that have a PruneFactor times higher score than the lowest scoring path (-PruneFactor X (inf))
AmScale="-AmScale ${Am_Scale}"                                                                   # acoustic model scaling factor (Parameter: -AmScale AcousticModelScalingFactor (1))

### Reading from HTK lattices (mostly just for conversion) ###
LatticeFileType='-LatticeFileType htk'                                                    # Format of lattice files (-LatticeFileType [cmu|htk|openfst] (text))
HTKLMScale='-HTKLMScale 0'
ReadNodeTimes='-ReadNodeTimes'

if [ ! -e "${Output_Directory_Basename}/KnownN_${Known_N}_UnkN_${Unk_N}/Sentences_Iter_${4}" ]; then
  ${latticewordsegmentation_bin} ${KnownN} \
                            ${UnkN} \
                            ${AddCharN} \
                            ${NoThreads} \
                            ${PruneFactor} \
                            ${InputFilesList} \
                            ${InputType} \
                            ${SymbolFile} \
                            ${Debug} \
                            ${LatticeFileType} \
                            ${ExportLattices} \
                            ${NumIter} \
                            ${OutputDirectoryBasename} \
                            ${OutputFilesBasename} \
                            ${ReferenceTranscription} \
                            ${CalculateLPER} \
                            ${CalculatePER} \
                            ${CalculateWER} \
                            ${SwitchIter} \
                            ${AmScale} \
                            ${InitLM} \
                            ${InitLmNumIterations} \
                            ${PruningStep} \
                            ${BeamWidth} \
                            ${OutputEditOperations} \
                            ${EvalInterval} \
                            ${WordLengthModulation} \
                            ${UseViterby} \
                            ${DeactivateCharacterModel} \
                            ${HTKLMScale} \
                            ${ReadNodeTimes}
else
  echo "Word segmentaiton has been run already. Skipping."
fi
