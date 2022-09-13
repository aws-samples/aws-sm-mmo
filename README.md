## Serve multiple models with One Sagamaker Endpoint 

1.	Launches an g4dn.2xlarge instance with Deep Learning AMI (Ubuntu 20.04)
2.	Clone this repo
```
git clone https://github.com/aws-samples/aws-sm-mmo
```
3. Extract model file
```
cd model/bert_base/1/
unzip model.bin.zip
```
4.	Build docker image
```
./build.sh mytriton
```
4.  Launch a notebook instance with ml.c5.xlarge specification from Sagemkaer console
5.  Upload mytriton.ipynb to notebook instance, then execute step by step.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

