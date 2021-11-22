# ccminer

suprminer sp-mod (september 2019) optimized x16r/x16rv2/x17 algo without any dev fee.


This miner is not updated anymore. I did a quick test of the x17 algo on my RTX 2060 SUPER. The latest ccminer 2.3 does around 9MHASH,The popular alexis 1.0.0 fork is doing around 12 MHASH My free opensource suprminer/SPMOD-GIT is doing 19MHASH. The private miner Enemy/T-rex & co is doing around 26MHASH.
But it all doesn't really matter because there is no profit in 26MHASH. 

Most optimizations come from sp, so please support him.

Overclock the core and memory for the best performance

This variant was tested and built on Linux (ubuntu server 14.04, 16.04, Fedora 22 to 25)
It is also built for Windows 7 to 10 with VStudio 2013, to stay compatible with Windows 7 and Vista.

Note that the x86 releases are generally faster than x64 ones on Windows, but that tend to change with the recent drivers.

The recommended CUDA Toolkit version is 9.2
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)
- Curl (prebuilt for win)
- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows


Compile on Linux
----------------

Please see [INSTALL](https://github.com/tpruvot/ccminer/blob/linux/INSTALL) file or [project Wiki](https://github.com/tpruvot/ccminer/wiki/Compatibility)
