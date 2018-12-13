function evaldetector(pos, neg, filename, overlap_thresh)
  neg = char(neg);

  [names, label, result, overlap, score] = textread(filename, '%s %d %d %f %f', 'delimiter' , ',', 'headerlines', 1);

  pos_index = find(strcmp(names, pos));
  pos_total = length(pos_index);
  total = length(result);

  tp = sum(result(overlap(pos_index) >= overlap_thresh));
  fp = sum(result(overlap(pos_index) < overlap_thresh));
  fn = sum(result(pos_index) == 0);

  fprintf('%s evaluation - total tested samples %d\n', pos, total);
  fprintf('Target is present [%s] - total samples %d, true positive: %d, false positive: %d, false negative: %d\n', pos, pos_total, tp, fp, fn);

  tn = 0;
  for i=1:size(neg, 1)
      name = strtrim(neg(i,:));
      neg_index = find(strcmp(names, name));
      neg_total = length(neg_index);
      fp_neg = sum(result(neg_index));
      tn_neg = neg_total-fp_neg;
      fp = fp + fp_neg;
      tn = tn + tn_neg;
      fprintf('Target is not present [%s] - total samples %d, true negative: %d false positive: %d\n', name, neg_total, tn_neg, fp_neg);
  end

  fprintf('Confusion Matrix - true positive: %d, false positive: %d, true negative: %d false negative: %d\n', tp, fp, tn, fn);

  evaldetector_plot(pos, label, result, overlap, score, overlap_thresh)
end

function evaldetector_plot(target_name, label, result, overlap, score, overlap_thresh)
  nc = length(score);
  target = zeros(1, nc);

  % true positive
  % positive sample
  % detector result is true
  % overlap bigger than threshold
  target(label==1 & result==1 & overlap>=overlap_thresh)=1;

  % false negative
  % positive sample
  % detector result is false
  target(label==1 & result==0)=1;

  % true negative
  % negative samples
  target(label==-1)=0;

  total_samples = ones(1, nc);

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
  title(sprintf('%s\nAverage Precision = %.3f',upper(target_name), ap), 'FontSize', 16);

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
  title(sprintf('%s\nArea Under ROC Curve = %.3f',upper(target_name), auc), 'FontSize', 16);
end