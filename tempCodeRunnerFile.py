accuracy = 100* total_correct/total
    end_time = time.time() - start_time

    # Printing out statistics
    print("Epoch no.",epoch+1 ,"|accuracy: ", round(accuracy, 3),"%", "|total_loss: ", total_loss, "| epoch_duration: ", round(end_time,2),"sec")