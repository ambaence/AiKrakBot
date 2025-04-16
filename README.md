# AiKrakBot

## File Structure
```
[01;34m.[00m
├── [01;34mbackend[00m
│   ├── [01;32malerts.py[00m
│   ├── [01;32mapi_handler.py[00m
│   ├── [01;32mbacktester.py[00m
│   ├── [01;32m__init__.py[00m
│   ├── [01;32mlogin.py[00m
│   ├── [01;34mml_engine[00m
│   │   ├── [01;32mactor_critic.py[00m
│   │   ├── [01;32mautoencoder.py[00m
│   │   ├── [01;32mdqn_model.py[00m
│   │   ├── [01;32mensemble.py[00m
│   │   ├── [01;32mgan_model.py[00m
│   │   ├── [01;32mgnn_model.py[00m
│   │   ├── [01;32mgru_model.py[00m
│   │   ├── [01;32mhrl_model.py[00m
│   │   ├── [01;32m__init__.py[00m
│   │   ├── [01;32mlstm_model.py[00m
│   │   ├── [01;32mppo_model.py[00m
│   │   ├── [01;32mrandom_forest.py[00m
│   │   ├── [01;32msentiment.py[00m
│   │   ├── [01;32mtcn_model.py[00m
│   │   ├── [01;32mtransfer_learning.py[00m
│   │   └── [01;32mtransformer_model.py[00m
│   ├── [01;34m__pycache__[00m
│   │   ├── alerts.cpython-38.pyc
│   │   ├── api_handler.cpython-312.pyc
│   │   ├── api_handler.cpython-38.pyc
│   │   ├── backtester.cpython-38.pyc
│   │   ├── __init__.cpython-312.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── login.cpython-38.pyc
│   │   ├── risk_management.cpython-38.pyc
│   │   └── security.cpython-38.pyc
│   ├── [01;32mrisk_management.py[00m
│   ├── [01;32msecurity.py[00m
│   └── [01;34mstrategies[00m
│       ├── [01;32marbitrage_strategy.py[00m
│       ├── [01;32mbreakout_strategy.py[00m
│       ├── [01;32mdca_strategy.py[00m
│       ├── [01;32mgrid_trading_strategy.py[00m
│       ├── [01;32m__init__.py[00m
│       ├── [01;32mmanager.py[00m
│       ├── [01;32mmean_reversion_strategy.py[00m
│       ├── [01;32mmomentum_strategy.py[00m
│       ├── [01;32mpair_trading_strategy.py[00m
│       └── [01;32mscalping_strategy.py[00m
├── [01;32mconfig.py[00m
├── .env
├── [01;32m.env.txt[00m
├── file_tree.txt
├── [01;34mfrontend[00m
│   ├── [01;32mapp.py[00m
│   └── [01;32m__init__.py[00m
├── [01;34m.git[00m
│   ├── [01;32mCOMMIT_EDITMSG[00m
│   ├── [01;32mconfig[00m
│   ├── [01;32mdescription[00m
│   ├── [01;32mHEAD[00m
│   ├── [01;34mhooks[00m
│   │   ├── [01;32mapplypatch-msg.sample[00m
│   │   ├── [01;32mcommit-msg.sample[00m
│   │   ├── [01;32mfsmonitor-watchman.sample[00m
│   │   ├── [01;32mpost-update.sample[00m
│   │   ├── [01;32mpre-applypatch.sample[00m
│   │   ├── [01;32mpre-commit.sample[00m
│   │   ├── [01;32mpre-merge-commit.sample[00m
│   │   ├── [01;32mprepare-commit-msg.sample[00m
│   │   ├── [01;32mpre-push.sample[00m
│   │   ├── [01;32mpre-rebase.sample[00m
│   │   ├── [01;32mpre-receive.sample[00m
│   │   ├── [01;32mpush-to-checkout.sample[00m
│   │   ├── [01;32msendemail-validate.sample[00m
│   │   └── [01;32mupdate.sample[00m
│   ├── [01;32mindex[00m
│   ├── [01;34minfo[00m
│   │   └── [01;32mexclude[00m
│   ├── [01;34mlogs[00m
│   │   ├── [01;32mHEAD[00m
│   │   └── [01;34mrefs[00m
│   │       └── [01;34mheads[00m
│   │           └── [01;32mmaster[00m
│   ├── [01;34mobjects[00m
│   │   ├── [01;34m0b[00m
│   │   │   └── [01;32mfaa27297af8d07ef5380cf3ddfb758d16cf942[00m
│   │   ├── [01;34m10[00m
│   │   │   └── [01;32m4f48e5eb459b4a1bd2839337b82b08a34d32b2[00m
│   │   ├── [01;34m15[00m
│   │   │   └── [01;32m74bff5ab9d4ea4da5ae9aeccaadeb29eeeac45[00m
│   │   ├── [01;34m17[00m
│   │   │   ├── [01;32ma53774b8b98c28aaa43d409062856e894fcfc1[00m
│   │   │   └── [01;32mb1749e57ba2fdbd72a10795f90d03c5be46d75[00m
│   │   ├── [01;34m1d[00m
│   │   │   └── [01;32mc3bd1240b399671e957b935e086283552123c2[00m
│   │   ├── [01;34m21[00m
│   │   │   └── [01;32m15ca26f14ecb1278851e4ad5b7f0c6f9bfcdff[00m
│   │   ├── [01;34m23[00m
│   │   │   └── [01;32mf25a7090278f6cc02e377c5ea36be038b445df[00m
│   │   ├── [01;34m2c[00m
│   │   │   └── [01;32m2db7e131795a92b41e37ac190a9a5d3641a76c[00m
│   │   ├── [01;34m32[00m
│   │   │   └── [01;32m6e5e24cbe0fbb9f5855404e4f7aad5a55d4644[00m
│   │   ├── [01;34m34[00m
│   │   │   └── [01;32ma3d3e3a8ebd431a95a683422cd540c17bc58cf[00m
│   │   ├── [01;34m35[00m
│   │   │   └── [01;32m29f9a10ae3c569a869873133b86d62bdbeccc9[00m
│   │   ├── [01;34m38[00m
│   │   │   ├── [01;32m7bbe2a55eb227414824d045fcbe4d1a2ec3b22[00m
│   │   │   └── [01;32mf4059b7106f1d1bdf439f0827d214e22763cfe[00m
│   │   ├── [01;34m39[00m
│   │   │   └── [01;32mb4b49b12465a3669ea7172bf1911cbc45c2fe0[00m
│   │   ├── [01;34m48[00m
│   │   │   └── [01;32m53fbe4d7684e2797bc031d1e93636ee1178d84[00m
│   │   ├── [01;34m56[00m
│   │   │   └── [01;32m77586db9269c370485a6dd30e108dff9ad4271[00m
│   │   ├── [01;34m5c[00m
│   │   │   └── [01;32m35dc0ae2df0669ca83a8b1b0fd7b348d9b6a08[00m
│   │   ├── [01;34m66[00m
│   │   │   └── [01;32mf5144e558fbb3d2960771753877565bd5b7891[00m
│   │   ├── [01;34m67[00m
│   │   │   └── [01;32m5c73109c88bf43cd97845bb298ad1dab9b6252[00m
│   │   ├── [01;34m68[00m
│   │   │   └── [01;32m06a00c39399aa5c88cbbdf4faac3c59a50833d[00m
│   │   ├── [01;34m69[00m
│   │   │   └── [01;32mabebf36eed5be5014c000a3e5bff6f2c6567c9[00m
│   │   ├── [01;34m70[00m
│   │   │   └── [01;32m0afd54fd7a4543dfdbfb5a2c4c48fc493fdc96[00m
│   │   ├── [01;34m73[00m
│   │   │   └── [01;32m4fae3ebc9d76b6e3888bfa62da38f83491bf6f[00m
│   │   ├── [01;34m75[00m
│   │   │   └── [01;32me6386dbb9fcabfd07be5703df8917073cd6db4[00m
│   │   ├── [01;34m83[00m
│   │   │   ├── [01;32m15d497c69ce6e321ccbb8ede8bc0669021abfe[00m
│   │   │   └── [01;32m2c5915168a4cf7a1cc51bc094a4d5fb68812c7[00m
│   │   ├── [01;34m89[00m
│   │   │   └── [01;32md52246ad51c23c1271c05b6af7531de1e060e8[00m
│   │   ├── [01;34m90[00m
│   │   │   └── [01;32m3d27465ab82e1e62ebdbbcee7229395c9a3646[00m
│   │   ├── [01;34m93[00m
│   │   │   └── [01;32m1ced443db0395d3135b2c7daa7a2798736a6d9[00m
│   │   ├── [01;34m9a[00m
│   │   │   └── [01;32me5c9dd6ab4e983ae250fa5105a7e461be98ab6[00m
│   │   ├── [01;34m9b[00m
│   │   │   └── [01;32m9ecbcd60920241ebdf8e79f51ca9ddefae4c69[00m
│   │   ├── [01;34m9c[00m
│   │   │   └── [01;32me4a731c1a43f1fa8ce966f54a2539cf0dacdc7[00m
│   │   ├── [01;34m9d[00m
│   │   │   └── [01;32m1dcfdaf1a6857c5f83dc27019c7600e1ffaff8[00m
│   │   ├── [01;34m9f[00m
│   │   │   └── [01;32m56d571ae5e51d351d06ede01fda73da3c77cf3[00m
│   │   ├── [01;34ma6[00m
│   │   │   └── [01;32m770c481d493756ba01eae96a4a23287b0ec380[00m
│   │   ├── [01;34ma8[00m
│   │   │   └── [01;32md9e498344e291281423562628e9865f72696bf[00m
│   │   ├── [01;34ma9[00m
│   │   │   └── [01;32mc60c9077d6aa19d55f6cd00ae077ac494312ca[00m
│   │   ├── [01;34mae[00m
│   │   │   └── [01;32ma209cd17dd0ed417d86bd2f1471f9cd436609d[00m
│   │   ├── [01;34mb5[00m
│   │   │   └── [01;32md2210d1b35534ebf0acc0b96d617adf8eb1234[00m
│   │   ├── [01;34mba[00m
│   │   │   └── [01;32m246d9b28bcecffdb31d862ae2c7559999ffbd8[00m
│   │   ├── [01;34mbb[00m
│   │   │   ├── [01;32m043571f9e664cf8ccd1e3852efe26c030ca75a[00m
│   │   │   └── [01;32m592229b24217269b43f3e592aac05aa581e9b4[00m
│   │   ├── [01;34mc3[00m
│   │   │   └── [01;32me3c1e0b7dac07509443fad7a27868d5647df9f[00m
│   │   ├── [01;34mcf[00m
│   │   │   └── [01;32me8facf3b5b23f366ef8b30962d8ec166fcb322[00m
│   │   ├── [01;34md0[00m
│   │   │   └── [01;32m29a2e5fce4004fef45c7a26d4fba4a21960c29[00m
│   │   ├── [01;34md2[00m
│   │   │   └── [01;32mc2e0767fe085224d7733fbc457198e50e31389[00m
│   │   ├── [01;34md7[00m
│   │   │   └── [01;32ma4f948cd94ffd6b9429e06ca1b4c50798db7ed[00m
│   │   ├── [01;34md8[00m
│   │   │   └── [01;32m0fd2771611087fae041d8f3a8173e6c69408ed[00m
│   │   ├── [01;34mdb[00m
│   │   │   └── [01;32me25a25c7995b426864cba5bb1073df535760f4[00m
│   │   ├── [01;34mdd[00m
│   │   │   └── [01;32mca7559787cfe75ecf5f2889517815eb318e706[00m
│   │   ├── [01;34me2[00m
│   │   │   └── [01;32m22aef3fd7ec04627c818dff96625d429ddbcc8[00m
│   │   ├── [01;34me6[00m
│   │   │   └── [01;32m9de29bb2d1d6434b8b29ae775ad8c2e48c5391[00m
│   │   ├── [01;34me7[00m
│   │   │   └── [01;32m07dda894af6db2e14a725fdb7c95af4bdbaba5[00m
│   │   ├── [01;34mf0[00m
│   │   │   └── [01;32m96db1d742855f79a9d5c82874e6aa90cdb4d9a[00m
│   │   ├── [01;34mf1[00m
│   │   │   └── [01;32mad27d2ae734f17da9aeacf6bdfafa7dd1f360c[00m
│   │   ├── [01;34mf3[00m
│   │   │   └── [01;32mf5f8a0fefe4fdd079e61e121897c3a62f544c2[00m
│   │   ├── [01;34mf4[00m
│   │   │   ├── [01;32m0afa048a8b32c0de4c246aa433cbf7d2d5de6d[00m
│   │   │   └── [01;32mbd6b58e982ea9119abc891759646a755efb060[00m
│   │   ├── [01;34mf8[00m
│   │   │   └── [01;32m95d877a68105b6f10511930fff222c6e29f981[00m
│   │   ├── [01;34mfb[00m
│   │   │   └── [01;32m2551e8c9e07db84f2f6612a4989b50b845c007[00m
│   │   ├── [01;34minfo[00m
│   │   └── [01;34mpack[00m
│   └── [01;34mrefs[00m
│       ├── [01;34mheads[00m
│       │   └── [01;32mmaster[00m
│       └── [01;34mtags[00m
├── .gitignore
├── [01;32m__init__.py[00m
├── [01;34mlogs[00m
│   ├── bot.log
│   ├── bot.odt
│   ├── [01;32mgan_metrics.log[00m
│   └── [01;32mprofile.log[00m
├── [01;32mmain.py[00m
└── [01;32msetup_aikrakbot.sh[00m
```
