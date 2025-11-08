## context
A comment by user rebel45 on reddit post https://www.reddit.com/r/AskAstrophotography/comments/18gq2fp/stacking_multiple_nights/?show=original.

## instructions

This might be a long one but I’ll explain in detail as best as I can.

    + First modify the OSC Pre_Processing.ssf script file. Open it with any text editor and delete the following lines which are towards the bottom.

. #Align lights register pp_light

.#Stack calibrated light to result.fit stack r_pp_light rej 3 3 -norm=addscale -output_norm -rgb_equal -out=result

.#flip if required load result mirrorx -bottom up save …result_$LIVETIME:%d$s

    + Save the script file as OSC_Pre_Processing_No_Stacking in the same folder as the rest of the scripts. The next time you load Siril it should show up under the scripts menu. The modified script will do all the calibration preprocessing etc but it will skip the registration and stacking process which you will do later. It won’t make a result file.

    + For each night put those corresponding files into its own folder and subfolder. For example, night 1 would have its own folder with the subfolders of lights, darks, biases and flats. Night 2 would also have its own folder and sub folders of lights, darks, biases and flats. And so forth.

    + Locate the night 1 directory and tell Siril where it is by using the house button.

    + Run the OSC_Pre_Processing_No_Stacking script for each individual night. Once Siril is done it should’ve created the pp_light files for you with no result file. Delete everything else that Siril created for that script routine. You won’t need the rest moving forward.

    + Run that same script as mentioned above for the rest of the nights that you might have. Each night should be processed individually using the modified script. Keep those pp_light files where they are.

    + After you’ve processed each night go to the “Conversion” tab in Siril and add the pp_light files. To add those pp_light files hit the “+” button and navigate to wherever those pp_light files are. Keep adding those pp_light files until all the nights have been added to your list. You don’t need to rename the files.

    + Give the sequence a name in the “Sequence name” field. Make sure the “Symbolic link” is checked. “Debayer” doesn’t need to be checked. Push the “Convert” button. It should only take a minute to add the files to the sequence.

    + Once the sequence is made go to the “Sequence” tab and make sure the newly created sequence is selected.

    + Go to the “Registeration” tab. Leave everything the default setting and hit “Go Register”. This might take a few hours depending on how many files you have.

    + After the registration is completed, go back to the “Sequence” tab and make sure the “r_xx” sequence is selected.

    + Next go to the “Stacking” tab. I usually leave most everything the default. For Pixel rejection I seem to have better results with Sigma Clipping. Make sure the “Output Normalization” and “RGB Equalization” is checked. Hit “Start Stacking”. This will also take a few hours depending on how many files you have. The result file it generates after stacking is what you use.

It’s not as automated as it could be but I work from home so I can monitor everything while I work. I’m not an expert by any means but this is working for me pretty well. 