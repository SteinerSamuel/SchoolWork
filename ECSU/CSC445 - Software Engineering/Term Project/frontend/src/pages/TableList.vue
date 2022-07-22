<template>
  <div class="content">
    <div class="md-layout">
      
      
      <div class="md-layout-item md-medium-size-100 md-xsmall-size-100 md-size-100" :key=componentKey>
        <md-card>
          <md-card-header data-background-color="blue">
            <h4 class="title">Results</h4>
            <div class="md-layout">
              <md-field class="md-layout-item md-medium-size-100 md-xsmall-size-100 md-size-50">
                <label>Search...</label>
                <md-input v-model="search" type="text" v-on:keyup.enter="searchAction"></md-input>
              </md-field>
              <div class="md-layout-item"></div>
              <div class="md-radio md-radio-inline">
                <input type="radio" id="Title" v-model="radio" value="name" name="Title" :key="name"/>
                <label for="Author"> <small> Title </small></label>
              </div>
              <div class="md-radio md-radio-inline">
                <input type="radio" id="Author" v-model="radio" value="author" name="Author" :key="author"/>
                <label for="Author"> <small> Author </small></label>
              </div>
              <div class="md-radio md-radio-inline">
                <input type="radio" id="ISBN" v-model="radio" value="ISBN" name="ISBN" :key="ISBN"/>
                <label for="ISBN"> <small> ISBN </small></label>
              </div>
              
              <div>
                <md-button class="md-dense md-round" @click="searchAction">Search</md-button>
              </div>
            </div>
          </md-card-header>
          <md-card-content >
            <simple-table tableHeaderColor="blue" :tableData=apiurl v-on:rented_book="componentKey++"></simple-table>
          </md-card-content>
        </md-card>
      </div>

    </div>
  </div>
</template>

<script>
import { SimpleTable } from "@/components";
export default {
  name: "RegularRadio",
  methods:{
    searchAction :function () {
        if  (this.$data.search !== null){ 
          if (this.radio == "name"){
            this.$data.apiurl = "http://127.0.0.1:5000/"+ this.radio +"/?name=" + this.$data.search;
            }
            else if (this.radio == "author"){
                this.$data.apiurl = "http://127.0.0.1:5000/"+ this.radio +"/?name=" + this.$data.search;
            }
            else if (this.radio == "ISBN"){
              this.$data.apiurl = "http://127.0.0.1:5000/"+ this.radio + "/?isbn=" + this.$data.search;
            }
        this.componentKey +=1;
        }
        else{
          this.$data.apiurl ="http://127.0.0.1:5000/name/";
          this.componentKey +=1;
        }
      
    }
  },
  data: () =>({
    search: null,
    radio: "name",
    apiurl: 'http://127.0.0.1:5000/name/',
    componentKey: 0

  }),
  components: {
    SimpleTable
  }
};
</script>
