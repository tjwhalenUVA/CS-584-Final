
#Classes Count====
genderCount <- 
  voice %>%
  mutate(count = 1) %>%
  select(count, gender=label) %>%
  gather(feature, value, -gender) %>%
  group_by(gender, feature) %>%
  summarise(value = sum(value)) %>%
  ggplot(aes(x=gender, 
             y=value, 
             fill=gender)) +
  geom_bar(stat = 'identity', 
           show.legend = F) +
  geom_text(aes(y=value/2, label=value), 
            fontface='bold') +
  theme_classic() +
  labs(title='Class Counts', 
       x='Gender', 
       y=NULL)

#Corr Plot====
cormat <- round(cor(voice[,1:20]),2)

melted_cormat <- melt(cormat)

reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

# Reorder the correlation matrix
cormat <- reorder_cormat(cormat)

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)

# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)

# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed() + 
  geom_text(aes(Var2, Var1, label = value), 
            color = "black", 
            size = 2) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")

#Density====
allDensity <-
  voice %>%
  gather(key="Measure", 
         value = "Value", 
         -label) %>%
  ggplot(., 
       aes(fill=label)) +
  geom_density(aes(x=Value), 
               alpha=0.4) +
  facet_wrap(~Measure, 
             scales = "free") +
  theme_classic() +
  theme(legend.position = "bottom", 
        axis.text = element_blank(), 
        axis.ticks = element_blank(), 
        axis.title = element_blank())

#BoxPlot====
allBoxPlot <-
  voice %>%
  gather(key="Measure", 
         value = "Value", 
         -label) %>%
  ggplot(., 
         aes(fill=label)) +
  geom_boxplot(aes(y=Value, x=label), 
               alpha=0.4) +
  facet_wrap(~Measure, 
             scales = "free") +
  theme_classic() +
  theme(legend.position = "bottom", 
        axis.text = element_blank(), 
        axis.ticks = element_blank(), 
        axis.title = element_blank())


#One Variable vs All====
oneVall <-
  voice %>%
  gather("Measure", 
         "Value", 
         -label, -meanfun) %>%
  ggplot(aes(x=meanfun, y=Value)) +
  geom_point(aes(color=label, fill='black'), alpha = 0.3) +
  facet_wrap(~Measure, 
             scales = "free_y") +
  theme_classic() +
  theme(legend.position = "bottom", 
        axis.text = element_blank(), 
        axis.ticks = element_blank())

#Sample Plot====
scatterSample <-
  voice %>%
  select(meanfun,
         IQR,
         Gender = label) %>%
  ggplot(aes(x=meanfun, y=IQR, color=Gender)) +
  geom_point() +
  theme_bw() +
  labs(x='Avg Fundamental Frequency Across Acoustic Signal',
       y='Interquantile Range (in kHz)')
  


#3D Plot====
p <- plot_ly(voice, 
             x = ~meanfun, 
             y = ~IQR, 
             z = ~sfm, 
             color = ~label) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'meanfun'),
                      yaxis = list(title = 'IQR'),
                      zaxis = list(title = 'sfm')))

#T Tests====
voiceCols <- names(voice %>% select(-label))

voiceSlim <-
  voice %>%
  gather(feature, value, -label)

finalDF <- data.frame()

for(i in voiceCols){
  #Run Test
  vtest <- t.test(filter(voiceSlim, feature == i)$value~voice$label, 
                  conf.level = 0.99)
  #Extract Values
  df <- NULL
  df$Feature <- i
  df$`T Value` <- vtest$statistic
  df$df <- vtest$parameter
  df$`P Value` <- vtest$p.value
  df$`CI Lower` <- vtest$conf.int[1]
  df$`CI Upper` <- vtest$conf.int[2]
  df$Female <- vtest$estimate[1]
  df$Male <- vtest$estimate[2]
  loopDF <- data.frame(df)
  #Combine Results
  finalDF <- bind_rows(finalDF, loopDF)
}

tTestDT <-
  finalDF %>%
  mutate(T.Value = round(T.Value, 3), 
         df = round(df, 1), 
         P.Value = round(P.Value, 5), 
         CI.Lower = round(CI.Lower, 3), 
         CI.Upper = round(CI.Upper, 3), 
         Female = round(Female, 3), 
         Male = round(Male, 3)) %>%
  datatable(., 
            rownames = F, 
            caption='Two Sample T-test. 99% Confidence Level', 
            options(paging=F, 
                    dom='t', 
                    order = list(list(3, 'asc'))
                    )) %>% 
  formatStyle(
    'P.Value',
    target = 'row',
    backgroundColor = styleInterval(0.01, c('lightgreen', 'gray'))
  )


#Summary====

