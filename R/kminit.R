kminit <- function(y,L,...,seed = sample.int(.Machine$integer.max, 1),cluster=FALSE){
  set.seed(seed)
  kmres <-kmeans(y,L,...)
  if(cluster){
    list(mean=t(kmres$centers),
         var=simplify2array(lapply(split(as.data.frame(y),kmres$cluster),var)),
         cluster=1*t(sapply(kmres$cluster,function(x){1:L %in% x})),
         kmres=kmres)
  }else{
    list(mean=t(kmres$centers),
         var=simplify2array(lapply(split(as.data.frame(y),kmres$cluster),var)),
         kmres=kmres)
  }
}