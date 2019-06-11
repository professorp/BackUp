bool test(lli n){
    if(n < 2) return false;
    if(n == 2) return true;
    lli d = n - 1, s = 0;
    while(!(d & 1)){
        d >>= 1;
        ++s;
    }
    for(int i = 0; i < 16; ++i){
        lli a = 1 + rand() % (n - 1);
        lli m = powMod(a, d, n);
        if(m == 1 || m == n - 1) goto exit;
        for(int k = 0; k < s - 1; ++k){
            m = m * m % n;
            if(m == n - 1) goto exit;
        }
        return false;
        exit:;
    }
    return true;
}