'''optimizer --> minimze the loss function, to do that, we need a learning rate'''
'''m--> slope b--> interferer'''
import tensorflow as tf 
import numpy as np
import tempfile


'''DEFINE COMO OS DADOS ESTAO ESTRUTURADOS'''

#nome das colunas
_CSV_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education','education_num', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']
#define o tipo de cada coluna --> 0=int, ''=str, nada []=float
_CSV_COLUMN_DEFAULTS =[[0], [''], [0], [''], [0], [''], [''], [''], [''], [''], [0], [0], [0], [''],['']]

#funcao que REFINA os dados
def input_fn(data_file, num_epochs, shuffle, batch_size):

#checa se o arquivo existe(opcional)
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

#cria funcao que DECODIFICA o csv(arquivo) baseado no _CSV_COLUMNS e no _CSV_COLUMNS_DEFAULTS (e ainda define os labels)
  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))#pega todas as e define elas como sendo as features(o que serve pra prever os resultados)
    labels = features.pop('income_bracket')#acontece que dentro das features tem a previsao(se recebe ou nao mais q 50k, entao .pop retira ela de lá e coloca dentro de labels(output)
    print(labels)
    print()
    return features, tf.equal(labels, '>50K') #retorna as features e e os labels(os labels foram 'traduzidos' para 0 ou 1, como boolean)

 #pega os dados por linhas
  dataset = tf.data.TextLineDataset(data_file)

  #if shuffle:  '''isso eu tirei pq parece bem inutil, nao mudou o resultado final e ia dar mais trabalho'''
  #  dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)
 
 #APLICA A FUNCAO PARSE_CSV que acabamos de criar PARA CADA ITEM EXISTENTE
  dataset = dataset.map(parse_csv, num_parallel_calls=5)


  dataset = dataset.repeat(num_epochs)#traducao que achei na internet --> One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.(tipo quantidade de vezes que ele vai passar pelos dados)
  dataset = dataset.batch(batch_size)#numero de exemplos de treinamento

#refina os negocio tudo com um iterator (que eh justamente um objeto que represeta um conjunto de dados)
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels

#define nossos parametros(eu escolhi 10 e 10 aleatoriamente, ainda to vendo as melhores opcoes
train_data = 'adult.csv'#dados que vao ser usados pra treinar
num_epochs = 10
batch_size = 10
test_data = 'adult.test'#dados que vao mostrar se ta funcionando direitinho

#DEFININDO COLUNAS DAS FEATURES // coloca as possiveis opcoes para cada coluna
#coluna pode ser com vocabularios(strings) ou com valores numericos mesmo, cada um tem uma funcao propia como pode ver
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])


#um com strings, mas mais flexivel:
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)

#education_x_occupation = tf.feature_column.crossed_column( '''voce pode juntar eles como se relacionassem intimamente, mas nao precisei'''
#    ['education', 'occupation'], hash_bucket_size=1000)

#aqui os com valores numericos
age = tf.feature_column.numeric_column('age')

age_buckets = tf.feature_column.bucketized_column(#esse aqui dovide em grupos(os que tem entre 18 e 25, emtre 60 e 65 e etc)
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

#colunas que serao realmente usadas
base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]
print(education)
#outras colunas usadas, mas que se relacionam intimamente (se a pessoa tiver mt educacao, mas pra uma profissao mal paga, ou vice versa, por isso se relacionam)
crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]
#DEFINIDO AS COLUNAS DAS FEATURES

#USANDO O MODEL, DANDO AS FEATURES
model_dir = tempfile.mkdtemp()#so cria um arquivo, nao precisa saber mt, eu acho
'''aqui onde tudo acontece, a gente junta tudo, onde estara o modelo(no arquivo que acabamos de criar, quais colunas usaremos
e ainda usamos otimizadores como learning rate(opcionais, ja tem alguns predefinidos)'''
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1.0, l2_regularization_strength=1.0))

#TREINO
model.train(input_fn=lambda:input_fn(train_data, num_epochs, True, batch_size))#treina esse modelo, que nos acabamos de dar os parametros, a partir dos dados que foram refinados com o input_fn

#TESTANDO
#results vai dar estatisticas de precisao e etc (A PRIMEIRA LINHA EH PRA DAR 83% E UNS QUEBRADO, DEU ISSO NO TUTORIAL, DEU ISSO NO MEU)
results = model.evaluate(input_fn=lambda: input_fn(
    test_data, 1, False, batch_size))
for key in sorted(results):
  print('%s: %s' % (key, results[key]))
print()
print()
helper = model.predict(input_fn=lambda: input_fn(test_data, 1, False, 1))#vamos prever se os nossos dados tao dando oq era pra dar
#so descomenta isso aqui em baixo pra ele printar todas as respostas pra tu comparar, se quiser cria um programa simples, tipo (numero de acertos/numero total) pra ver se eh bom
#for predict in helper:
#	print(predict['classes'])