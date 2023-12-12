\subsection{Edge Cases}\label{subsec:edge-cases}
The model evaluation based on the validation set demonstrates two crucial findings.
Firstly, adding L2 regularisation to the models yields significant improvement when it comes to dealing with overfitting. Figure @Training and Validation Loss over Epochs@ shows the training and validation loss with and without L2 regularisation for model @model@.
Secondly, the L2 regularisation improves performance, especially in later epochs, compared to the models without L2 regularisation. Figure @Validation ROC AUC  over Epochs-without L2@ and @Validation PR AUC over Epochs-with L2@ provide an overview of the validaiton results measured by these metrics.

Overall, the models based on melspectograms seem to perform better, according to ROC AUC and PR AUC. 

Predicting the labels on the test test yields similar, but slightly differnt results, as shown in Table @Comparison ROC-PR-Loss@. The performance difference between models with and without L2 regularizaton is less significant. Based on the evaluation results of the test set, focusing on ROC AUC, PR AUC, two models seem to be the ones performing best.
'trainer_mel7_l2' has the highest ROC AUC and PR AUC, which indicates it is the best at distinguishing between classes and maintaining a balance between precision and recall.
'trainer_mel4_l2' has a slightly lower ROC AUC than trainer_mel7_l2 but higher than the other models, and it has a relatively high PR AUC, making it a strong contender as well.
Overall, 'trainer_mel7_l2' seems to perform best of the custom-mode CNNs, considering only these 3 metrics.

The edge case analysis is based on the evaluation results of predicting the 50 labels of the 4328 samples in the test set, using model 'trainer_mel7_l2'.

To find and evaluate this model's edge cases, the samples and the labels can be analyzed.

\subsubsection{Samples}\label{subsec:samples}
To analyze the edge cases based on the samples, two approaches were undertaken.
The first one is to calculate the subset accuracy, which is a rather strict approach, as as a sample is only considered to be correctly classified if all predicted labels match the true labels. In this case, 91.17 % of the samples are considered to be misclassfied, which means, that at least one of the predicted labels does not match the true labels of the respective samples.
The second approach is to calculate the Jaccard Similarity. It is used to identify samples where the model's prediction significantly diverges from the actual labels. Samples with a Jaccard Similarity below a certain threshold could be potential edge cases. The average Jaccard Similarity is 0.36 with 22.87 % of the samples having a similarity of less than 0.01, which means that there is very little to no overlap between the predicted and true labels.
The third approach is to calculate the sample-wise Precision, Recall, and F1 Score. The averaga F1 Score is 0.48, average Precision, 0.61, and average Recall 0.45. Since precision, reacall, and F1 result in the same set of samples after applying the threshold, the focus lies on the F1 Score for finding the edge cases. A low or zero F1 Score indicates that the model failed to correctly predict most or any of the relevant labels for those samples.
It makes sense, that the Jaccard Similarity and F1 Score are quite similar,  because both metrics require at least one correct label match to be greater than zero, and a complete absence of correct predictions means there are no true positives to contribute to either score.
This means, combining Jaccard similarity and F1 Score results in the same 22.87 % of misclassfied labels. As a result, these can be considered as samples that are edge cases.
The next step would be to conduct a detailed analysis of those samples identified as edge cases, meaning, e.g., analysing the waveforms or spectograms used as input, and listening to the audio clips. Due to the vast number of labels identified as edge cases (990), this was considered to be out of scope of this work.
It should be mentioned at this point, that one common reason for sample edge cases lies in the difference of length in the audio clips used as input. To avoid this, it was made sure during preprocessing to trim all audio length to the same length of 29.1 seconds.

\subsubsection{Labels}\label{subsec:labels}
Proceeding with the edge case analysis, the attention lies now on the labels. 

Finding edge cases in the labels breaks down to finding labels the evaluated model finds hard to predict. To find these labels, two approaches were undertaken. 

The first one is to calculatae the Hammming Loss, which is used to understand which labels are most frequently misclassified across all samples. Labels with a high contribution to the Hamming Loss could be a sign for potential edge cases. For the 50 labels, the average Hamming Loss is 0.05, with 18.00 % of the labels having a hamming loss above 0.1, which is nearly double the average. This results in the following 9 labels being potential edge cases based on Hamming Loss: 'guitar', 'classical', 'slow', 'string', 'vocal', 'electro', 'drum', 'no singer', and 'fast'.

The second approach, similar to finding the sample edge cases, is to calculate the label-wise F1 Score. The averaga F1 Score is 0.31, average Precision, 0.46, and average Recall 0.28. Since precision, reacall, and F1 result in the same set of labels after applying the threshold, the focus lies on the F1 Score for finding the edge cases. 24.00 % of the labels are below the threshold for an F1 Score of 0.01, which means that the model failed to predict the label in most or all of the samples. Such labels are 'no singer', 'solo', 'new age', 'strange', 'bass', 'modern', 'no piano', 'baroque', 'foreign', 'trance', 'folk', and 'no beat' in this case. As a result, these labels represent potential edge cases.

Combining Hamming Loss and F1 Score, results in one label crossing the threshold of of both, which is 'no singer'.

\subsubsection{Interpretation}\label{subsec:interpretation}
To further investigate the label edge cases, it makes sense to analyze the distribution of music tracks by labels in the training data. Figure @Labels by Track Number and Scores@ shows the distribution, with colorized bars showing the indicating which labels where identified as edge cases based on Hamming Loss, F1 Score, and a combination of both.
Interesstingly, it seems that the labels with a low F1 Score are more on the side of the labels with less tracks, while the labels with a high Hamming Loss are primarily the ones with more tracks per label in the training data.
Imbalance in the dataset can be a major reason for edge cases, which is why it makes sense, that labels the model finds hard to predict, are the ones underrepresented in the training data, as this does not give the model as much training on identifying these labels as it gets for the labels with more tracks. This is roughly confirmed by the distribution of the labels with low F1 Scores.
Meanwhile, the label edge cases identified by the high Hamming Loss are more on the side of the labels with more training data. This seems a bit counterintuitive at first, but there could be several potential reasons for this. Labels with a high occurence in the training data could be hard to identify for the model because they might cover a wide range of styles, which makes it harder for the model to learn how to identify them. Also, the model might be getting too fixated on the common patterns and missing out on the finer details of these labels, leading to mistakes. Addittionally, there could be an underlying issue with the way the labels were assigned to the data initially.
Lastly, the label 'no singer' has both, a high contribution to Hamming Loss, and a low F1 Score. As a result this is an edge case that the model has a prticulary hard time to identify. As for the high Hamming Loss and low F1 Score labels, there could be several potential reasons for that. Potentially, the characteristics of having no vocals in a track  is especially challenging for the model to learn due to subtle audio features.

Overall, several edge cases were identified and qunatitatively and qualitatively analyzed, based on the samples and based on the labels. There are several reasons for why the model does not perform well in these cases. To further deal with them, a more detailed analysis of the data is needed. Based on these findings, the model's performance could potentially be improved by adjusting the preprocessing, form of input, model architecture, hyperparameter tuning, or other factors. Due to this work's limited scope this will be left unexplored for now.
