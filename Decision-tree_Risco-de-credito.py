
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


base = pd.read_csv('risco_credito.csv')


# In[3]:


previsores = base.iloc[:, 0:4].values


# In[4]:


classe = base.iloc[:, 4].values


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


labelencoder = LabelEncoder()


# In[7]:


previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])


# In[14]:


from sklearn.tree import DecisionTreeClassifier, export


# In[11]:


classificador = DecisionTreeClassifier(criterion='entropy')


# In[12]:


classificador.fit(previsores, classe)


# In[15]:


# Verificando a importância de cada atributo
print(classificador.feature_importances_)


# In[17]:


# Criando arquivo para visualização gráfica da árvore de decisão criada
export.export_graphviz(classificador,
                       out_file='arvore.dot',
                       feature_names=['historia', 'divida', 'garantias', 'renda'],
                       class_names=['alto', 'moderado', 'baixo'],
                       filled=True,
                       leaves_parallel=True)


# In[18]:


resultados = classificador.predict([[0,0,1,2], [3,0,0,0]])


# In[19]:


resultados


# In[22]:


print(classificador.classes_)

