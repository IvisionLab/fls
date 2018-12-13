clear
close all

% M = csvread('../assets/matches/gemini_resnet50_deltas20181205T1744_matches.csv');
M = csvread('../assets/matches/mask_rcnn_gemini_resnet50_20181207T1921_matches.csv');


target = M(:,1);
score = M(:,2);
overlap= M(:,3);
ious= M(:,4);
target(overlap < 0.5) = 0;

nc = length(score);
% nc = 10000
target = target';
total_samples = ones(1, length(score));

% generate thresholds
qvals = (1:(nc-1))/nc;
thresh = [min(score), quantile(score,qvals)];
thresh = sort(unique(thresh),2,'descend');

total_pos = sum(target);
total_neg = sum(total_samples - target);

nt = length(thresh);
tpr = zeros(1, nt);
fpr = zeros(1, nt);
prec = zeros(1, nt);


for i=1:nt
  idx = score >= thresh(i);
  fpr(i) = sum(total_samples(idx) - target(idx));
  tpr(i) = sum(target(idx)) / total_pos;
  prec(i) = sum(target(idx)) / sum(total_samples(idx));
end

% recall is equal to true positive rate
rec=tpr;
size(rec)
% false positive rate
fpr = fpr / total_neg;

% calculate the average precision
mrec=rec';
mpre=prec';
for i=numel(mpre)-1:-1:1
  mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
fprintf('Average Precision = %.3f\n',ap);

% plot precision recall curve
figure;
plot(tpr, prec,'-', 'linewidth', 2);
grid;
xlabel('Recall', 'FontSize', 16);
ylabel('Precision', 'FontSize', 16);
title(sprintf('Average Precision = %.3f', ap), 'FontSize', 16);

% calculate the roc curve
base = abs(fpr(2:end)-fpr(1:end-1));
height_avg = (tpr(2:end)+tpr(1:end-1))/2;
auc = sum(base .* height_avg);
fprintf('Area Under ROC Curve = %.3f\n',auc);

% plot roc curve
figure;
plot(fpr, tpr,'-', 'linewidth', 2);
grid;
xlabel('False Positive Rate', 'FontSize', 16);
ylabel('True Positive Rate', 'FontSize', 16);
title(sprintf('Area Under ROC Curve = %.5f', auc), 'FontSize', 16);