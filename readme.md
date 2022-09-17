# *Soji DNN Dev's Repo*

Hola, programador! Este repositorio de GitHub es donde estaremos compartiendo codigo y donde estare subiendo programas de ejemplo para que vean como se hacen las cosas.

### Para los de comunicaciones:

Actualmente estoy usando una libreria llamada Socket. Esta libreria permite comunicarme de manera rapida entre 2 computadoras bajo la misma red. Es muy rapido, pero es facil de causar errores, lo que puede interrumpir el robot en medio de su operacion. 

> Como primer proyecto, quiero que comuniquen 2 computadoras entre si mismas. Esto lo pueden hacer utilizando el mismo Socket, pero prefiero si investigan de otros metodos de hacer esa comunicacion.
> Luego de ese proyecto, vamos a mejorar, arreglar, actualizar y eficientizar la comunicacion sea adentro de Socket, o con una libreria nueva. Suerte!

### Para los de Robot:

En este repositorio hay un folder con el nombre de pi_internals. En este folder esta el codigo que yo escribi para controlar el prototipo -1. Aqui, hay varios programas pero el principal es botware.py
Ese programa lo estaremos usando como una libreria en nuestro programa principal, para controlar toda la parte del robot. Seleccionare a los 3 mejores programadores para trabajar conmigo en el programa principal, y tal vez en el de AI.

Para ustedes, hay 4 trabajos que pueden hacer:
> Convertir coordenadas (X, Y, direccion) en velocidades para ambas ruedas (Buscar Mecanismo Diferencial)

> Adaptar el codigo actual para controlar el controlador de motores a ser utilizado: Cytron 10A Dual DC Motor Driver (Buscar en google para pinout)

> Leer un codificador de motor y determinar que distancia ha viajado en una cierta cantidad de pulsos, donde los datos de entrada son (pulsos_por_rev, entrada_de_pulsos, diametro), y para conseguir la distancia deben calcular la circunferencia (diametro*pi)

# *~PARA TODOS~*:

Para poder hacer lo pedido, primero deben investigar sobre la libreria utilizada en el caso especifico. Los 2 mejores recursos para ello son: 
> Buscar la libreria en google y leer la documentacion (Como si fuera un datasheet)
> Buscar la libreria en PyPI, ir a su pagina principal, y buscar ejemplos en youtube (Manera Preferida por mi)

Si necesitan ayuda en algo, por favor diganme que estoy mas que feliz en ayudarlos. No soy profesor ni voy a ponerles nota. Solo quiero que aprendan esta herramienta increiblemente util, y que disfruten el proceso.

**-GG**
