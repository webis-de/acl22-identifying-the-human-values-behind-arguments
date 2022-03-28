library(rlang)

##### User Interface #####

args <- commandArgs(trailingOnly=TRUE)
rscript.options <- commandArgs(trailingOnly = FALSE)
source.dir <- dirname(sub(".*=", "", rscript.options[grep("--file=", rscript.options)]))

help_text <- paste(
  '\nUsage:  Evaluation.R [OPTIONS]',
  '',
  'Evaluate the specified predictions in regards to Precision, Recall, F1 and Accuracy.',
  'The scores are calculated for each model every label individually with the mean score for each level.',
  '',
  'Options:',
  '  -a, --absent-labels       Include absent labels from the test dataset into validation',
  '  -d, --data-dir string     Directory with the prediction and argument files (default',
  '                            WORKING_DIR/webis-argvalues-22/)',
  '  -h, --help                Display help text',
  '  -p, --predictions string  Predictions file with predictions to evaluate (default',
  '                            WORKING_DIR/predictions.tsv)',
  sep = '\n'
  )
exit <- function() { invokeRestart("abort") }

# default values
data_dir <- file.path('./webis-argvalues-22/')
prediction_filepath <- file.path('./predictions.tsv')
absent_labels <- FALSE

i <- 1
while (i <= length(args)) {
  arg <- args[i]
  if (arg == "-a" || startsWith("--absent-labels", arg)) {
    absent_labels <- TRUE
  } else if (arg == "-d" || startsWith("--data-dir", arg)) {
    if (i == length(args)) {
      stop("No data directory specified", call. = FALSE)
    }
    dir <- file.path(args[i + 1])
    i <- i + 1
  } else if (arg == "-h" || startsWith("--help", arg)) {
    cat(help_text)
    exit()
  } else if (arg == "-p" || startsWith("--predictions", arg)) {
    if (i == length(args)) {
      stop("No predictions file specified", call. = FALSE)
    }
    prediction_filepath <- file.path(args[i + 1])
    i <- i + 1
  } else {
    stop(help_text, call. = FALSE)
  }
  i <- i + 1
}

arguments_filepath <- file.path(data_dir, 'arguments.tsv')
if (!file.exists(prediction_filepath)) {
  stop("The specified prediction file does not exist.", call. = FALSE)
}

levels <- c("1", "2", "3", "4a", "4b")

##########################


##### Load components #####

source(paste(source.dir, 'components/Metrics.R', sep="/"))

###########################


##### Setup #####

`%notin%` <- Negate(`%in%`)

cat('===> Loading files...\n')

data.arguments <- read.csv(arguments_filepath, sep = '\t', stringsAsFactors = FALSE)
data.predictions <- read.csv(prediction_filepath, sep = '\t', stringsAsFactors = FALSE)

data.arguments.filtered <- data.arguments[data.arguments$Argument.ID %in% data.predictions$Argument.ID,]
if ('Usage' %in% colnames(data.arguments)) {
  data.arguments.filtered <- data.arguments.filtered[data.arguments.filtered$Usage == "test",]
}
# remove un-predictable arguments
data.predictions <- data.predictions[data.predictions$Argument.ID %in% data.arguments.filtered$Argument.ID,]

has.parts <- TRUE
has.methods <- TRUE

# store usable levels
actual.levels <- c()

data.labels.all <- list()
for (i in 1:length(levels)) {
  label.file.path <- file.path(data_dir, paste('labels-level', levels[i], '.tsv', sep = ''))
  if (file.exists(label.file.path)) {
    read_labels <- read.csv(label.file.path, sep = '\t', stringsAsFactors = FALSE)
    read_labels <- read_labels[read_labels$Argument.ID %in% data.arguments.filtered$Argument.ID,]
    read_labels <- read_labels[,names(read_labels) %in% names(data.predictions)]
    if (length(names(read_labels)) > 1) {
      data.labels.all[[levels[i]]] <- read_labels
      actual.levels <- c(actual.levels, levels[i])
    }
  }
  else {
    cat(paste('No file for level ', levels[i], ' found.\n', sep = ''))
  }
}

levels <- actual.levels

if (length(levels) == 0) {
  cat("For all levels the required files were either absent or don't apply on the predictions. No evaluation can be made.\n")
  exit()
}

if ('Part' %in% colnames(data.arguments)) {
  datasetNames <- unique(data.arguments.filtered$Part)
  
  data.predictions$dataset <- sapply(data.predictions$Argument.ID, function(x) {
      c(data.arguments.filtered[data.arguments.filtered$Argument.ID == x,"Part"])
    })
} else {
  datasetNames <- c("none")
  data.predictions$dataset <- c("none")
  has.parts <- FALSE
}

if ('Method' %notin% colnames(data.predictions)) {
  data.predictions$Method <- 'Bert'
  has.methods <- FALSE
}

data.methods <- split(data.predictions, data.predictions$Method)

#################


##### Prepare labels #####

if (!absent_labels) {
  absent.label.list <- list()
  for (i in 1:length(datasetNames)) {
    absent.labels <- c()
    for (j in 1:length(levels)) {
      
      active.labels <- data.labels.all[[levels[j]]]
      active.labels <- active.labels[active.labels$Argument.ID %in% (
        data.predictions[data.predictions$dataset == datasetNames[i],"Argument.ID"]
        ),]
      active.labels$Argument.ID <- NULL
      
      cSums <- colSums(active.labels)
      absent.labels <- c(absent.labels, names(cSums)[which(cSums == 0)])
    }
    if (!is_empty(absent.labels)) {
      absent.label.list[[datasetNames[i]]] <- absent.labels
    }
  }
}

##########################


##### Execute evaluation #####

cat('===> Evaluating predictions...\n')

data.evaluation <- NULL

for (i in 1:length(levels)) {
  data.labels <- data.labels.all[[levels[i]]]
  for (j in 1:length(data.methods)) {
    data.method.datasets <- split(data.methods[[j]], (data.methods[[j]])$dataset)
    for (k in 1:length(data.method.datasets)) {
      dataPred <- data.method.datasets[[k]]
      active.dataset <- names(data.method.datasets)[k]
      if (!absent_labels && !is.null(absent.label.list[[active.dataset]])) {
        dataPred <- dataPred[,names(dataPred) %notin% absent.label.list[[active.dataset]]]
      }
      dataResult <- evaluate(dataPred, data.labels, names(data.methods)[j], active.dataset, levels[i])
      
      data.evaluation <- rbind(data.evaluation, dataResult)
    }
  }
}

##############################


##### Visual formatting #####

## Special cases
if (is.element("1", actual.levels)) {
  data.evaluation$Label[data.evaluation$Label == "Be.self.disciplined"] <- "Be self-disciplined"
}
if (is.element("2", actual.levels)) {
  data.evaluation$Label[data.evaluation$Label == "Self.direction..thought"] <- "Self-direction: thought"
  data.evaluation$Label[data.evaluation$Label == "Self.direction..action"] <- "Self-direction: action"
  data.evaluation$Label[data.evaluation$Label == "Power..dominance"] <- "Power: dominance"
  data.evaluation$Label[data.evaluation$Label == "Power..resources"] <- "Power: resources"
  data.evaluation$Label[data.evaluation$Label == "Security..personal"] <- "Security: personal"
  data.evaluation$Label[data.evaluation$Label == "Security..societal"] <- "Security: societal"
  data.evaluation$Label[data.evaluation$Label == "Conformity..rules"] <- "Conformity: rules"
  data.evaluation$Label[data.evaluation$Label == "Conformity..interpersonal"] <- "Conformity: interpersonal"
  data.evaluation$Label[data.evaluation$Label == "Benevolence..caring"] <- "Benevolence: caring"
  data.evaluation$Label[data.evaluation$Label == "Benevolence..dependability"] <- "Benevolence: dependability"
  data.evaluation$Label[data.evaluation$Label == "Universalism..concern"] <- "Universalism: concern"
  data.evaluation$Label[data.evaluation$Label == "Universalism..nature"] <- "Universalism: nature"
  data.evaluation$Label[data.evaluation$Label == "Universalism..tolerance"] <- "Universalism: tolerance"
  data.evaluation$Label[data.evaluation$Label == "Universalism..objectivity"] <- "Universalism: objectivity"
}
if (is.element("3", actual.levels)) {
  data.evaluation$Label[data.evaluation$Label == "Self.enhancement"] <- "Self-enhancement"
  data.evaluation$Label[data.evaluation$Label == "Self.transcendence"] <- "Self-transcendence"
}
if (is.element("4b", actual.levels)) {
  data.evaluation$Label[data.evaluation$Label == "Growth..Anxiety.free"] <- "Growth, Anxiety-free"
  data.evaluation$Label[data.evaluation$Label == "Self.protection..Anxiety.avoidance"] <- "Self-protection, Anxiety avoidance"
}

## Revert spaces
data.evaluation$Label <- sapply(data.evaluation$Label, function(x) gsub('\\.', ' ', x))

## Round metric values
data.evaluation$Precision <- sapply(data.evaluation$Precision, function(x) round(x, digits = 2))
data.evaluation$Recall <- sapply(data.evaluation$Recall, function(x) round(x, digits = 2))
data.evaluation$F1 <- sapply(data.evaluation$F1, function(x) round(x, digits = 2))
data.evaluation$Accuracy <- sapply(data.evaluation$Accuracy, function(x) round(x, digits = 2))

## Reorder rows and finish up
if (has.parts) {
  new_order <- sapply(c(datasetNames), function(x){which(data.evaluation$`Test dataset` == x)})
  data.evaluation <- data.evaluation[unlist(new_order),]
} else {
  data.evaluation$`Test dataset` <- NULL
}

if (has.methods) {
  new_order <- sapply(names(data.methods), function(x){which(data.evaluation$Method == x)})
  data.evaluation <- data.evaluation[unlist(new_order),]
} else {
  data.evaluation$Method <- NULL
}

#############################


##### Output final evaluation #####
write.table(data.evaluation, stdout(), sep = '\t', row.names = F, quote = F, col.names = T)

###################################
