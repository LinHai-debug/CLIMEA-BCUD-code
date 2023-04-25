library(reticulate)
library(stringi)
library(MBC)

np <- import("numpy")
Pool <- import("multiprocessing")
os <- import("os")
sys <- import("sys")
warnings <- import("warnings")

warnings$filterwarnings("ignore")

is_leap <- function(year){
    if (year%%4==0){
        if (year%%100==0){
            if (year%%400==0){return (1)}else{return (0)}            
        }else{return (1)}
    }else{return (0)}
}

find_filename <- function(root_dir,year1,year2){
    files <- sort(os$listdir(root_dir))
    file_list = list()
    for (file in files){
    if (as.numeric(substring(file,1,4)) >= year1 & as.numeric(substring(file,1,4)) <= year2){
        # if (is_leap(as.numeric(substring(file,1,4))) & 
            # as.numeric(substring(file,5,7)) == 60 & 
            # grepl("MSWX", root_dir)){next}#365 need next######################################
           file_list <- c(file_list,stri_c(root_dir,file))
        }
    }
    return (file_list)
}

find_filename_dates <- function(root_dir,year1,year2){
    files <- sort(os$listdir(root_dir))
    file_list = list()
    for (file in files){
    if (as.numeric(substring(file,1,4)) >= year1 & as.numeric(substring(file,1,4)) <= year2){
        # if (is_leap(as.numeric(substring(file,1,4))) & 
            # as.numeric(substring(file,5,7)) == 60 & 
            # grepl("MSWX", root_dir)){next}#365 need next########################################
           file_list <- c(file_list,file)
        }
    }
    return (file_list)
}

read_npy_batch <- function(root_dir,year1,year2){
    file_list <- find_filename(root_dir,year1,year2)
    print(stri_c('data_list: ',length(file_list)))
    data = np$zeros(c(length(file_list),55L,85L))
    i <- 1
    for (file_name in file_list){
        if (grepl('MSWX',file_name)){
            data[i,,] <- np$flipud(np$squeeze(np$load(file_name)))
            }else{
            data[i,,] <- np$squeeze(np$load(file_name))
            }
        i <- i+1
        }
    # print(dim(data))
    # print(class(data))
    # print(np$mean(data))
    return (data)
}


variables <- list('variables you want to apply QDM')##########################################################################
for (variable in variables){
    obs_dir <- stri_c('OBS path')
    if (variable == 'TEM'){v <- 'tas'}
    else if (variable == 'MAX'){v <- 'tasmax'}
    else if (variable == 'MIN'){v <- 'tasmin'}
    else if (variable == 'WIN'){v <- 'sfcWind'}
    else if (variable == 'PRE'){v <- 'pr'}
    else if (variable == 'LWD'){v <- 'rlds'}
    else if (variable == 'SWD'){v <- 'rsds'}
    else if (variable == 'SHU'){v <- 'huss'}
    else if (variable == 'RHU'){v <- 'hurs'}

    model_names <- list('model you want to apply QDM')

    args <- commandArgs(TRUE) 
    ssp <- args[3]
    method <- args[5]
    wet_day <- args[4]
    start <- as.numeric(args[1])
    end <- as.numeric(args[2])

    for (model_name in model_names){
        mod_dir <- stri_c('mod path')
        sce_dir <- stri_c('sce path')
        dates <- find_filename_dates(sce_dir,start,end)
        save_dir <- stri_c('save path')
        if (!os$path$exists(save_dir)){os$makedirs(save_dir)}
        
        obs_data <- read_npy_batch(obs_dir,1999,2014)
        mod_data <- read_npy_batch(mod_dir,1999,2014)
        sce_data <- read_npy_batch(sce_dir,start,end)
        
        data_adj <- np$zeros(c(dim(sce_data)[1],dim(sce_data)[2],dim(sce_data)[3]))
        for (i in 1:dim(sce_data)[2]){
            for (j in 1:dim(sce_data)[3]){
                fit.qdm <- QDM(o.c=obs_data[,i,j],m.c=mod_data[,i,j],
                               m.p=sce_data[,i,j],ratio=FALSE)###############################################
                data_adj[,i,j] <- fit.qdm$mhat.p
                #print(np$mean(data_adj))
            }
        }
        i <- 1
        for (date in dates){
            np$save(stri_c(save_dir,date),data_adj[i,,])
            i <- i+1
        }
        print(stri_c('dates_len: ',length(dates)))
        print(stri_c('dim_sce_data: ',dim(sce_data)))
        print(stri_c(model_name,' Done!'))
    }
}