require(rjson)
require(e1071)
require(pracma)
require(lomb)
require(fractal)
require(stringr)

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

Rprof()

df = data.frame()
i <- 0
for (jsonfile in dir(basedir, pattern = 'deviceMotion_walking_outbound.json.item*', recursive=TRUE)){
  #print(jsonfile)

  json_data <- fromJSON(file=paste0(basedir, jsonfile))

  json_data <- lapply(json_data, function(x) {
    x[sapply(x, is.null)] <- NA
    unlist(x)
  })

  json_data <- as.data.frame(do.call("rbind", json_data))
  tst <- trim_data(json_data)
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


  i <- i+1



  if(i == 4) {
    break
  }

}

summaryRprof()