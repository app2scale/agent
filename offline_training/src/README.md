Evolution ile belli aralıklarla yapılabilir. Mesela her 2 iterayonda bir yapılıp onun metrikleri alınabilir.
Çalışırken tensorboard ile inceleyebiliriz.
Save and load policy. Checkpoint olabilir.
tune.fit methodu ile de çağırabiliriz. Daha hızlı çalışabilir
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py ile callback yazılıp metrikler çekilebilir.
Evolution ayarlandıktan sonra uzun training'e (100, 1000) bırakılabilir.  https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-evaluation-options
Parameter optimization için grid search bak.
Policy oluşturduktan sonra bir tane test env olacak. Action kısmı random olmayacak. policy test edilecek.



Experiences
Sadece cost kullanılarak reward tanımı yaptığımızda agent bütün kaynakları minimum düzeyde kullanmayı tercih ediyor. Sürekli re heap cpu azalatacak hamleler yapıyor.

out_tps 0, 1 normalizasyonu yapılıp reward fonksiyonunu out_tps-cost olarak tanımladığımızda agent re sayısını azaltıp cpu arttıracak şekilde hareket ediyor. 1,500,900 state'inden sonra hiçbir şey yapmamayı tercih ediyor.

out_tps ve cost zero mean unit variance normalizasyonu yapılınca 1 700 900 state'ine converge oluyor. Reward'ın maks olduğu 1 700 900 statinde kalıyor.

