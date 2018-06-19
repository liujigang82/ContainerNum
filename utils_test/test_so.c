//cppclTest.c
#include <stdio.h>
#include <dlfcn.h>
#include <string>
#include<python3.6/Python.h>
using namespace std;
int main() {
    typedef int(*container_rec)(int a, int b);
    void *handle;
    dlopen("libpython3.6m.so.1.0", RTLD_LAZY | RTLD_GLOBAL);
    handle = dlopen("./add.so", RTLD_LAZY);
    if(!handle)
    {
	    printf("ERROR_____");
            printf("ERROR, Message(%s).\n", dlerror());
            return -1;
    }

    container_rec my_container_rec = (container_rec)dlsym(handle, "add");
    char* szError = dlerror();
    if(szError != NULL)
    {
        printf("ERROR, Message(%s).\n", szError);
        dlclose(handle);
        return -1;
    }


    int result = my_container_rec(1, 2);

    dlclose(handle);

    printf("%d\n", result);
}