<template>
  <div class="content">
    <div class="md-layout">
      <div class="md-layout-item md-medium-size-100 md-xsmall-size-100 md-size-100" :key=componentKey>
        <md-card>
          <md-card-header data-background-color="orange">
            <h4 class="title">Rentals</h4>
          </md-card-header>
          <md-card-content >
            <ordered-table tableHeaderColor="orange" :tableData="this.apiurl + this.$keycloak.userName" v-on:returned_book="componentKey++">></ordered-table>
          </md-card-content>
        </md-card>
      </div>

    </div>
  </div>
</template>

<script>
import { OrderedTable } from "@/components";
export default {
  name: "RegularRadio",
  data: () =>({
    search: null,
    radio: "name",
    apiurl: ('http://127.0.0.1:5000/rented/?user='),
    componentKey: 0

  }),
  components: { 
    OrderedTable
  }
};
</script>
