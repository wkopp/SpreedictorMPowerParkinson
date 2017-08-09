require(rjson)
require(e1071)
require(pracma)
require(lomb)
require(fractal)
require(stringr)

library(foreach)
library(doParallel)

#setup parallel backend to use many processors
cores=detectCores()
cl <- makeCluster(cores[1]-1) #not to overload your computer
registerDoParallel(cl)




source('commonFunctions.R')
source('walkingModule.R')

basedir = '../../../data/download/'

trim_data <- function(M) {
    M['timestamp'] <- M['timestamp'] - M[1,'timestamp']
    M["squaresum"] = sqrt(M["userAcceleration.x"]**2 + M["userAcceleration.y"]**2 + M["userAcceleration.z"]**2)
    # take the threshold to be the extreme outliers
    threshold = quantile(M[,"squaresum"], 0.75)*3 - quantile(M[,"squaresum"], 0.25)*2

    idx <- which(M[,"squaresum"] >= threshold)
    idx <- idx[1]:idx[length(idx)]
    Mtrimmed = M[idx,]
    return(Mtrimmed)
}

json_files <- dir(basedir, pattern = 'deviceMotion_walking_outbound.json.item*', recursive=TRUE)

#Rprof()
#i <- 0


json_files_to_process<-json_files
#system.time(
#for (jsonfile in json_files_to_process){
features <- foreach(i=1:length(json_files_to_process), .combine=rbind) %dopar% {

  require(rjson)
  require(e1071)
  require(pracma)
  require(lomb)
  require(fractal)
  require(stringr)

  #print(jsonfile)

  df = data.frame()


  jsonfile <- json_files_to_process[i]

  json_data <- fromJSON(file=paste0(basedir, jsonfile))

  if(length(json_data) > 0) {
    json_data <- lapply(json_data, function(x) {
      x[sapply(x, is.null)] <- NA
      unlist(x)
    })

    json_data <- as.data.frame(do.call("rbind", json_data))


    #tst <- trim_data(json_data)
    tst <- json_data

    tst$squaresum = sqrt(tst["userAcceleration.x"]**2 + tst["userAcceleration.y"]**2 + tst["userAcceleration.z"]**2)

    diffsquares <- sqrt(diff(as.matrix(tst["userAcceleration.x"]))**2
                        + diff(as.matrix(tst["userAcceleration.y"]))**2
                        + diff(as.matrix(tst["userAcceleration.z"]))**2)

    for (var in c('userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z', 'squaresum')) {
      stats <- SingleAxisFeatures(tst[[var]], tst$timestamp, varName='')
      stats["variable"] <- var
      stats["filename"] <- jsonfile
      stats["fileid"] <- as.numeric(str_match(jsonfile, pat='\\d+/(\\d+)/')[,2])
      df <- rbind(df, stats)
    }

    stats <- SingleAxisFeatures(diffsquares, tst$timestamp[2:length(tst$timestamp)], varName='')
    stats["variable"] <- 'diffsquares'
    stats["filename"] <- jsonfile
    stats["fileid"] <- as.numeric(str_match(jsonfile, pat='\\d+/(\\d+)/')[,2])
    df <- rbind(df, stats)
  }


  df

}
#)
#Rprof(NULL)
#summaryRprof()

#stop cluster
stopCluster(cl)

