
err_num=0;
for i=1:num

    this_num = committee_judge_batch(batchdata(:,:,i),batchtargets(:,:,i),c_W1_cl,c_W2_cl,c_W3_cl,c_W4_cl,cate,boost_factor);
    
    err_num=err_num+this_num;
    

end
    
error_rate=err_num/(volu*num);


[volu,dim,test_num]=size(testbatchdata);

test_err_num=0;
for i=1:test_num

    this_num = committee_judge_batch(testbatchdata(:,:,i),testbatchtargets(:,:,i),c_W1_cl,c_W2_cl,c_W3_cl,c_W4_cl,cate,boost_factor);
    
    test_err_num=test_err_num+this_num;
    

end
    
test_error_rate=test_err_num/(volu*test_num);






