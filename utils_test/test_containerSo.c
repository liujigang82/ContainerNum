#include "Python.h"
int main(int argc, char** argv)
{
    // 初始化Python
    //在使用Python系统前，必须使用Py_Initialize对其
    //进行初始化。它会载入Python的内建模块并添加系统路
    //径到模块搜索路径中。这个函数没有返回值，检查系统
    //是否初始化成功需要使用Py_IsInitialized。
    Py_Initialize();

    // 检查初始化是否成功
    if ( !Py_IsInitialized() ) {
        return -1;
    }
    // 添加当前路径
    //把输入的字符串作为Python代码直接运行，返回0
    //表示成功，-1表示有错。大多时候错误都是因为字符串
    //中有语法错误。
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("print ('---import sys---')");
    PyRun_SimpleString("sys.path.append('./')");
    PyObject *pName,*pModule,*pDict,*pFunc;

    // 载入名为numrec的脚本
    pName = PyUnicode_FromString("numrec");
    pModule = PyImport_Import(pName);
    if ( !pModule ) {
        printf("can't find numrec.so");
        getchar();
        return -1;
    }
    pDict = PyModule_GetDict(pModule);
    if ( !pDict ) {
        return -1;
    }

    // 找出函数名为num_rec的函数
    pFunc = PyDict_GetItemString(pDict, "num_rec");
    if ( !pFunc || !PyCallable_Check(pFunc) ) {
        printf("can't find function [add]");
        getchar();
        return -1;
     }
    PyObject *pArg = NULL, *result = NULL;
    pArg = Py_BuildValue("(s)", "./img2/BMOU.jpg");
    result = PyEval_CallObject(pFunc, pArg);
    char *myString; 
    PyArg_Parse(result, "s", &myString);
    printf("%s\n", myString);

    Py_DECREF(pName);
    Py_DECREF(pModule);

    // 关闭Python
    Py_Finalize();
    return 0;
}
