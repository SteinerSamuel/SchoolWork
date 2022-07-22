<template>
  <div>
    <md-dialog-confirm 
      :md-active.sync="confirmDialog" 
      :md-content="confirm_title"
      md-click-outside-to-close=false
      md-close-on-esc=false
      md-title="Confirm Rental"
      @md-cancel="showDialog=true, confirmDialog=false"
      @md-confirm="rent()">
    </md-dialog-confirm>
    <md-dialog :md-active.sync="showDialog" :key=componentKey>
      <md-dialog-title class='md-layout md-gutter'>
        <div class ="md-layout-item">{{BookInfoData[0].Title}} 
          <br/><h5> By {{BookInfoData[0].Author}}</h5>
        </div>

      <i v-on:click="showDialog=false" class="fas fa-times"></i> &nbsp; &nbsp;&nbsp;&nbsp; &nbsp;
      </md-dialog-title>
      <md-dialog-content class="md-layout md-gutter">
        <div class="md-layout-item">
          <h5><b>DESCRIPTION:</b></h5>
          {{BookInfoData[0].Description}}
          <h5><b>PAGES:</b></h5> 
          {{BookInfoData[0].Pages}}
          <h5><b> PUBLICATION INFO: </b></h5>
          published {{BookInfoData[0].PublicationYear}} by {{BookInfoData[0].Publisher}} 
        </div>
        <div class="md-layout-item">
          <img style="width=100px" class="md-medium-size-25" :src="BookInfoData[0].CoverURL"/>
        </div>    
        
      </md-dialog-content>
      <md-dialog-content class="md-layout">
      <h5>Available Copies</h5>
      <md-table class="md-layout-item md-size-small-100" v-model=available_books :key=componentKey>
          <md-table-row @click="rent_confirm(item.Inventory_Id, $keycloak.userName, item.Title)" slot="md-table-row" slot-scope="{ item }">
          <md-table-cell>{{item.Title}}</md-table-cell>
          <md-table-cell>{{item.Author}}</md-table-cell>
          </md-table-row>
        </md-table>
      </md-dialog-content>
  </md-dialog>
    
    <md-table v-model=BookData :table-header-color="tableHeaderColor">
      <md-table-row @click="info(item.ISBN, item.Available)" slot="md-table-row" slot-scope="{ item }" :key=componentKey>
        <md-table-cell md-label="Title">{{ item.Title }}</md-table-cell>
        <md-table-cell md-label="Author">{{ item.Author }}</md-table-cell>
        <md-table-cell v-if="showAvailable" md-label="Available">{{ item.Available }}</md-table-cell>
        <md-table-cell v-if="showAvailable" md-label="Total">{{item.Total}}</md-table-cell>
      </md-table-row>
    </md-table>
  </div>
</template>

<script>
import axios from "axios";
export default {
  name: "simple-table",
  props: {
    tableHeaderColor: {
      type: String,
      default: ""
    },
    tableData: {
      default: "http://127.0.0.1:5000/"
    },
    showAvailable:{
      type: Boolean,
      default: true
    }
  },
  methods: {
    info :function (isbn, available){
      this.showDialog = true;
      this.book_avail = available;
      axios.get("http://127.0.0.1:5000/info/?isbn=" + isbn).then(response => (this.BookInfoData = response.data));
      axios.get("http://127.0.0.1:5000/available/?isbn=" + isbn).then(response => (this.available_books =response.data));

    },
    rent_confirm :function(Inventory_Id, user, title){
      this.invid = Inventory_Id;
      this.user = user;
      this.confirm_title = "Confirm rental of " + title;
      this.showDialog = false;
      this.confirmDialog = true;
    },
    rent :function(){
      axios.get("http://127.0.0.1:5000/rent_book/?id=" + this.invid + "&user=" + this.user);
      axios.get(this.tableData).then(response => (this.BookData = response.data.inventory));
      this.$emit('rented_book');
      this.showDialog = true;
      this.confirmDialog = false;
    }
  },
  data() {
    return {
      book_avail: 0,
      selected: [],
      BookData: null,
      available_books: null,
      BookInfoData: "Test",
      showDialog: false,
      componentKey: 0,
      confirmDialog: false,
      invid: "_",
      user: "_",
      confirm_title: "Test"
    };
  },
  
  mounted() {
    axios
      .get(this.tableData)
      .then(response => (this.BookData = response.data.inventory));
  }
};
</script>
