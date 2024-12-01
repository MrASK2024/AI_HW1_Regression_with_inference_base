# Анализ и предсказание цен на автомобили

## 1. Введение
В этом проекте был проведен анализ данных о ценах на автомобили и построение моделей для предсказания цен.

## 2. EDA анализ
1. Датасет разделен на тренировочный и тестовый:
   - Тренировочный: 6999 объектов и 13 признаков.
   - Тестовый: 1000 объектов и 13 признаков.
2. Имеются пропуски в столбцах:
   - mileage
   - engine
   - max_power
   - torque
   - seats
3. Числовые признаки содержат единицы измерения, из-за чего столбцы имеют тип `object`. Столбец `seats` имеет тип `float`, хотя количество мест всегда целое число.
4. Дубликаты в датасете:
   - Общее количество дубликатов: 1159 объектов.

## 3. Предобработка данных
1. Были удалены дубликаты и заполнены пропуски в столбцах медианным значением.
2. Столбец `torque` был разделен на два: `torque_nm` и `torque_rpm`.
3. Числовые признаки приведены к следующим типам данных:
   - `mileage`, `max_power`, `torque`: float
   - `engine`, `seats`: int

## 4. Статистический анализ
После предобработки данных были построены таблицы с основными статистическими характеристиками по числовым и категориальным столбцам. Данные представлены в файле `AI_HW1_Regression.ipynb`.

## 5. Визуализация данных
1. Построены pairplots, которые помогли предположить связь признаков с целевой переменной.
2. Связь с целевой перемнной прослеживается:
   - `year`, `max_power`, `engine`, `torque_nm`.
3. Положительная корреляция наблюдается между:
   - `engine`, `max_power` и `torque_nm`
4. Удалось увидеть, что при разделении на тест и трейн графики оказались похожими
5. Была построена таблица корреляции с коэффициентами Пирсона и тепловая карта. Выдвинутые гипотезы подтвердились. Наименее скоррелированы между собой `engine` и `year`.
   - Сильная положительная линейная зависимость наблюдается между `max_power` и `selling_price`, `engine` и `max_power`, `engine` и `seats`
   - Утверждать, что чем меньше год, тем, скорее всего, больше километров проехала машина к дате продажи можно, так как на данных прослеживается отрицательная корреляция между year и km_driven, хотя на графике практически не прослеживается эта зависимость

## 6. Скаттер-график
Построен скаттер-график между `selling_price` и `owner`. Было важно посмотреть взаимосвязь selling_price с owner(количество владельцев авто), как правило в жизни имеет место быть такая зависимость. В дальнейшем можно было преобразовать эти признаки one_hot кодированием и посмотреть как изменятся метрики модели.

## 7. Обучение модели линейной регрессии
1. Обучены модели линейной регрессии на данных:
   - Без нормализации и с нормализацией признаков.
2. Результаты:
   - **Без нормализации**:
     - MSE_train = 115621991307.98964
     - R2_train = 0.5966276110503121
     - MSE_test = 231752847054.7187
     - R2_test = 0.5968313599220589
   - **С нормализацией**:
     - MSE_train = 115621991307.98964
     - R2_train = 0.5966276110503121
     - MSE_test = 231752847054.7192
     - R2_test = 0.596831359922058

## 8. Регуляризация
1. Обучена модель линейной регрессии с L1 регуляризацией (Lasso).
2. Результаты:
   - MSE_train = 115621991318.08136
   - R2_train = 0.596627611015105
   - MSE_test = 231753527871.32025
   - R2_test = 0.5968301755400465

## 9. ElasticNet
1. Перебором по сетке с 10-ю фолдами был подобран оптимальный параметр alpha. Подбор проводился в диапазоне значений от 0.0001 до 1. Наилучший параметр alpha=0.22229964825261933. Качество модели по сравнению с предыдущими моделями ухудшилось.  
2. Результаты:
   - MSE_train = 117341062635.88867
   - R2_train = 0.5906302579476275
   - MSE_test = 245615723291.51776
   - R2_test = 0.5727148192582048

## 10. One-hot кодирование Ridge модель
1. Проведено one-hot кодирование категориальных столбцов. Качество модели с one_hot кодирование значительно улучшилось. Без него практически не изменилось.
2. Результаты:
   - **С one-hot кодированием**:
     - MSE_train = 94967472965.10231
     - R2_train = 0.6686853771579946
     - MSE_test = 209240301623.11884
     - R2_test = 0.635995290124858
   - **Без one-hot**:
     - MSE_train = 115621998955.96173
     - R2_train = 0.5966275843687031
     - MSE_test = 231774766919.7351
     - R2_test = 0.5967932270478282

## 11. Бизнес метрики
Оценка бизнес-метрик (доля предсказанных цен в пределах 10%):
- Linear регрессия: 232
- Lasso регрессия: 232
- Elastic регрессия: 242
- Ridge регрессия: 233

## 12. Реализация сервиса
Разработан сервис на FastAPI с функциями для:
- Предсказания цены одного автомобиля.
- Предсказания цен на список автомобилей.
Ссылка для просмотра [скринкаста работы сервиса](https://drive.google.com/file/d/1C23kyUm_P4-C5q5JaVyp0pV6GqEYvuA7/view?usp=sharing) 
  
## 13. Что не успел((:
К сожалению очень хотел, но катастрофически не хватает времени и работать и учиться, поэтому не успел сделать следующие моменты:
1. Прогнать все модели с добавлением one-hot кодированием категоральных признаков.
2. На мой взгляд было бы логичным пропуски в столбце seats не просто заменить на общую медиану по всем объектам, а разбить  на группы по name и там взять медиану. Потому что для определенной марки машины свои медианные значения мест. Возможно это просто мой бзик..
3. Создать пайплайны для преобразования признаков.
4. Проверить сервис на всех обученных моделях.
5. Реализовать валидацию входных данных в сервисе.
