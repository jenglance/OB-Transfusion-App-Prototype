library(shiny)
library(caret)
library(xgboost)
library(ggplot2)
library(dplyr)
library(pROC)
library(DT)

# ---- Load model bundle ----
model_bundle <- readRDS("xgb_model_bundle.rds")

xgb_model <- model_bundle$model
roc_obj_train <- model_bundle$roc
optimal_threshold <- model_bundle$best_threshold
threshold_metrics <- model_bundle$threshold_metrics
metrics <- model_bundle$metrics
factor_levels <- model_bundle$factor_levels

# ---- Extract factor levels for select inputs ----
indications_levels <- factor_levels$Cesarean.Indications

# ---- Binary choices for UI ----
binary_choices <- c("No" = "0", "Yes" = "1")

# ---- Risk band function ----
getRiskBand <- function(prob, threshold, buffer = 0.05) {
  if (prob >= (threshold + buffer)) {
    return("High Risk")
  } else if (prob <= (threshold - buffer)) {
    return("Low Risk")
  } else {
    return("Moderate Risk")
  }
}

# === UI ===
ui <- fluidPage(
  titlePanel("Transfusion Prediction App (XGBoost)"),
  
  tabsetPanel(
    tabPanel("Clinician",
             sidebarLayout(
               sidebarPanel(
                 selectInput("cesarean", "Cesarean Delivery:", choices = binary_choices),
                 selectInput("induction", "Induction Used:", choices = binary_choices),
                 selectInput("indications", "Indications:", choices = indications_levels),
                 actionButton("predictBtn", "Predict"),
                 hr(),
                 strong(paste("Optimal cutoff threshold (Youden's J):", round(optimal_threshold, 4))),
                 br(),
                 sliderInput("threshold", "Adjust Classification Threshold:", min = 0, max = 1,
                             value = optimal_threshold, step = 0.01),
                 sliderInput("buffer", "Moderate Risk Buffer (±):", min = 0, max = 0.1,
                             value = 0.05, step = 0.01),
                 hr(),
                 h4("Prediction Output:"),
                 verbatimTextOutput("predictionOutput")
               ),
               mainPanel(
                 h4("Feature Importance (by Gain):"),
                 plotOutput("importancePlot")
               )
             )
    ),
    
    tabPanel("Researcher",
             sidebarLayout(
               sidebarPanel(
                 fileInput("csvFile", "Upload CSV File", accept = ".csv"),
                 downloadButton("downloadResults", "Download Predictions"),
                 hr(),
                 h4("Classification Threshold"),
                 sliderInput("threshold", "Adjust Classification Threshold:", min = 0, max = 1,
                             value = optimal_threshold, step = 0.01),
                 sliderInput("buffer", "Moderate Risk Buffer (±):", min = 0, max = 0.1,
                             value = 0.05, step = 0.01)
               ),
               mainPanel(
                 h4("ROC Curve & AUC"),
                 plotOutput("rocPlot"),
                 verbatimTextOutput("aucOutput"),
                 hr(),
                 h4("Uploaded Data Predictions"),
                 DTOutput("uploadedPreds")
               )
             )
    )
  )
)

# === Server ===
server <- function(input, output, session) {
  
  # --- Reactive: preprocess uploaded CSV safely ---
  uploaded_data <- reactive({
    req(input$csvFile)
    
    tryCatch({
      df <- read.csv(input$csvFile$datapath, stringsAsFactors = TRUE)
      
      # --- Force factor levels and handle missing ---
      for(f in names(df)){
        if(is.factor(df[[f]]) || is.character(df[[f]])){
          df[[f]] <- addNA(df[[f]])
          levels(df[[f]])[is.na(levels(df[[f]]))] <- "7"
          df[[f]][is.na(df[[f]])] <- "7"
        }
      }
      
      # --- Force numeric/integer NA to 0 ---
      num_cols <- sapply(df, is.numeric) | sapply(df, is.integer)
      df[num_cols] <- lapply(df[num_cols], function(x){ x[is.na(x)] <- 0; x })
      
      # --- Ensure Cesarean.Indications exists ---
      if(!"Cesarean.Indications" %in% names(df)){
        df$Cesarean.Indications <- factor(rep("7", nrow(df)), levels = indications_levels)
      } else {
        df$Cesarean.Indications <- factor(df$Cesarean.Indications, levels = indications_levels)
      }
      
      # --- Add churn column if missing ---
      if(!"churn" %in% names(df)){
        if("Units" %in% names(df)){
          df$churn <- factor(ifelse(df$Units==0, "no","yes"), levels=c("no","yes"))
        } else {
          # If no target info, create dummy churn (will not affect ROC fallback)
          df$churn <- factor(rep("no", nrow(df)), levels=c("no","yes"))
        }
      }
      
      # --- Ensure all model columns exist ---
      model_cols <- names(xgb_model$trainingData)
      model_cols <- model_cols[model_cols != ".outcome"]
      missing_cols <- setdiff(model_cols, names(df))
      if(length(missing_cols) > 0){
        for(col in missing_cols) df[[col]] <- 0
      }
      df_model <- df[, model_cols, drop = FALSE]
      
      # --- Predict probabilities using trained model ---
      df$Probability_Yes <- predict(xgb_model, newdata = df_model, type = "prob")[,"yes"]
      
      return(df)
      
    }, error = function(e){
      message("Error processing uploaded CSV: ", e$message)
      return(NULL)
    })
  })
  # --- Download predictions for uploaded CSV ---
  output$downloadResults <- downloadHandler(
    filename = function() {
      paste0("predictions_", Sys.Date(), ".csv")
    },
    content = function(file) {
      df <- uploaded_data()
      
      # If uploaded CSV is NULL, use a dummy default with training model probabilities
      if(is.null(df)){
        # Optionally, fallback to training data or just return empty
        df <- data.frame(Note = "No uploaded CSV; cannot compute predictions")
      }
      
      write.csv(df, file, row.names = FALSE)
    }
  ) 
  # --- Update indications dynamically ---
  observe({
    req(input$cesarean)
    choices <- indications_levels
    if(input$cesarean=="1") choices <- setdiff(choices, "Vaginal")
    else if(!("Vaginal" %in% choices)) choices <- c("Vaginal", choices)
    selected_val <- input$indications
    if(is.null(selected_val) || !(selected_val %in% choices) || input$cesarean=="0") selected_val <- "Vaginal"
    updateSelectInput(session, "indications", choices=choices, selected=selected_val)
  })
  
  # --- Clinician predictions ---
  observeEvent(input$predictBtn, {
    tryCatch({
      new_data <- data.frame(
        CesareanDelivery = as.numeric(input$cesarean),
        InductionUsed = as.numeric(input$induction),
        Cesarean.Indications = factor(input$indications, levels = indications_levels),
        id = 9999
      )
      new_data <- new_data[, names(xgb_model$trainingData)[names(xgb_model$trainingData) != ".outcome"]]
      pred_prob <- predict(xgb_model, newdata=new_data, type="prob")[,"yes"]
      pred_class <- factor(ifelse(pred_prob >= input$threshold,"yes","no"), levels=c("no","yes"))
      risk_band <- getRiskBand(pred_prob, input$threshold, buffer=input$buffer)
      
      output$predictionOutput <- renderPrint({
        cat("Predicted Class:", as.character(pred_class), "\n")
        cat("Probability of Requiring a Transfusion:", round(pred_prob,4), "\n")
        cat("Risk Band:", risk_band, "\n")
        cat("Classification threshold used:", round(input$threshold,4), "\n")
        cat("Moderate Risk buffer (±):", round(input$buffer,4))
      })
    }, error=function(e){ 
      output$predictionOutput <- renderPrint({cat("Error in prediction:\n", e$message)}) 
    })
  })
  ###load library(dplyr) FOR FEATURE TITLE FIX IN R script!!!
  library(dplyr)
  # --- Feature importance CHECK COLUMN NAME DROP FACTORLEVEL name---
  output$importancePlot <- renderPlot({
    importance <- xgb.importance(model=xgb_model$finalModel) %>% filter(Feature!="id")%>%
    mutate(Feature = gsub("^Cesarean.Indications[._]?", "", Feature))
    ggplot(importance, aes(x=reorder(Feature, Gain), y=Gain)) +
      geom_bar(stat="identity", fill="lightblue") +
      coord_flip() +
      labs(title="Feature Importance by Gain", x="Feature", y="Gain") +
      theme_minimal()
  })
  
  # --- ROC plot ---
  output$rocPlot <- renderPlot({
    df <- NULL
    if(!is.null(input$csvFile)) df <- uploaded_data()
    
    if(!is.null(df) && all(c("churn","Probability_Yes") %in% colnames(df)) &&
       length(unique(df$churn)) > 1){
      roc_obj <- pROC::roc(df$churn, df$Probability_Yes)
      plot.roc(roc_obj, col="darkblue", main="ROC Curve (Uploaded Data)", print.auc=TRUE)
    } else {
      # Always fallback to training ROC at app start or if upload invalid
      plot.roc(roc_obj_train, col="darkblue", main="ROC Curve (Training CV - XGBoost Model Bundle)", print.auc=TRUE)
      if(!is.null(input$csvFile)){
        text(0.5,0.5,"Uploaded CSV invalid or missing 'churn'; showing training ROC", cex=1.2)
      }
    }
  })
  
  # --- AUC text ---
  output$aucOutput <- renderPrint({
    df <- NULL
    if(!is.null(input$csvFile)) df <- uploaded_data()
    
    if(!is.null(df) && all(c("churn","Probability_Yes") %in% colnames(df)) &&
       length(unique(df$churn)) > 1){
      roc_obj <- pROC::roc(df$churn, df$Probability_Yes)
      cat("AUC (Uploaded Data):", round(pROC::auc(roc_obj),4))
    } else {
      cat("AUC (Training CV - XGBoost Model Bundle):", round(pROC::auc(roc_obj_train),4))
    }
  })
}
# === Run app ===
shinyApp(ui=ui, server=server)